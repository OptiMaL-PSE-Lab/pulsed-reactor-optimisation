import jax.numpy as np
from flax.training import train_state
import optax
import jax
import flax.linen as nn
from reactor_synthesis import plot_design
import matplotlib.pyplot as plt 
from tqdm import tqdm 


def moving_averages(lst, window_sizes):
    assert all(window_size > 0 for window_size in window_sizes), "Window sizes should be greater than 0."
    assert len(lst) >= max(window_sizes), "Length of input list should be at least as large as the largest window size."

    result = {}
    for window_size in window_sizes:
        result[window_size] = [sum(lst[i-window_size+1:i+1])/window_size for i in range(window_size-1, len(lst))]
    return result



positive_samples = np.load("generative_design/2d.npy", mmap_mode="r")[:1000]
batch_key = jax.random.PRNGKey(seed=0)
negative_samples = jax.random.randint(batch_key,shape=(len(positive_samples),9,93),minval=0,maxval=2) 
d_in = np.concatenate([positive_samples,negative_samples],axis=0)
d_out = np.concatenate([np.ones(len(positive_samples),dtype='int32'),np.zeros(len(positive_samples),dtype='int32')],axis=0)
d = {'reactors':d_in,'label':d_out}

# shuffle both arrays in the same order
rng = jax.random.PRNGKey(seed=1)
rng, key = jax.random.split(rng)
shuffle = jax.random.permutation(key, len(d_in))
d_in = d_in[shuffle]
d_out = d_out[shuffle]
d = {'reactors':d_in,'label':d_out}

train_test_split = 0.8 
d_train = {'reactors':d_in[:int(train_test_split*len(d_in))],'label':d_out[:int(train_test_split*len(d_in))]}
d_test = {'reactors':d_in[int(train_test_split*len(d_in)):],'label':d_out[int(train_test_split*len(d_in)):]}

class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = np.expand_dims(x, axis=-1)  # Add channel dimension
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        # x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        # x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        # x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(3,3))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)
        return x

root_key = jax.random.PRNGKey(seed=0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

def create_train_state(rng):
  """Creates initial `TrainState`."""
  discriminator = Discriminator()
  params = discriminator.init(rng, np.ones([1,9,93]))['params']
  tx = optax.sgd(1e-3,0.9)
  return train_state.TrainState.create(
      apply_fn=discriminator.apply, params=params, tx=tx)

state = create_train_state(main_key)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            x=batch["reactors"],
        )
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["label"])
        loss = np.mean(loss)
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss,logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    predictions = logits > 0.5
    accuracy = np.mean(predictions==batch['label'])
    metrics = {
    'loss': loss,
      'accuracy': accuracy,
    }
    return state,metrics



def train_epoch(state, d, batch_size, epoch, rng):
  train_ds_size = len(d['reactors'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(d['reactors']))
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  batch_metrics = []

  for perm in tqdm((perms),position=0,leave=True):
    batch = {k: v[perm, ...] for k, v in d.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)
    tqdm.write('loss: %.4f, accuracy: %.2f' % ( metrics['loss'], metrics['accuracy'] * 100))

  training_batch_metrics = jax.device_get(batch_metrics)
  training_epoch_metrics = {
      k: np.mean(np.array([metrics[k] for metrics in training_batch_metrics]))
      for k in training_batch_metrics[0]}

  print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

  return state, training_epoch_metrics

@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  predictions = logits > 0.5
  accuracy = np.mean(predictions==labels)
  return grads, loss, accuracy


epochs = 100
batch_key = jax.random.PRNGKey(seed=8)
n_batch = 64
# initialise empty loss store 
losses = []
accs = []
test_losses = []
test_accs = []
for i in range(epochs):
    state, metrics = train_epoch(state, d_train, n_batch, i, batch_key)
    batch_key, subkey = jax.random.split(batch_key, 2)
    loss = metrics['loss'] 
    acc = metrics['accuracy']
    losses.append(loss)
    accs.append(acc)

    metric_lists = {'loss':losses,'accuracy':accs}
    
    _, test_loss, test_accuracy = apply_model(state, d_test['reactors'],
                                              d_test['label'])
    
    print('Test loss:',test_loss, ' Test accuracy:',test_accuracy)
    test_losses.append(test_loss)
    test_accs.append(test_accuracy) 

    test_metric_lists = {'loss':test_losses,'accuracy':test_accs}

    m = [metric_lists,test_metric_lists]
    if i > 10 and i % 5 == 0:
        fig,ax = plt.subplots(1,2,figsize=(10,3))
        ti = 0 
        ls = ['solid',':']
        names = ['Training','Test']
        for metrics in m:
            j = 0 
            for k,v in metrics.items():
                mv = moving_averages(v,[5])
                ax[j].plot(np.arange(len(v)),v,c='k',alpha=0.25,lw=1,ls=ls[ti],label=names[ti]+' '+k)
                ax[j].plot(np.arange(len(v))[4:],mv[5],alpha=0.75,ls=ls[ti],c='tab:blue',lw=2,label=names[ti]+' Moving average (5)')
                ax[j].spines['top'].set_visible(False)
                ax[j].spines['right'].set_visible(False)
                ax[j].set_xlabel('Epoch')
                ax[j].set_ylabel(k)
                ax[j].legend(frameon=False)
                j += 1
            ti += 1
        fig.tight_layout()
        fig.savefig('generative_design/loss.png')
        plt.close()

    # test_loss = compute_loss(state, test_batch)
    # print('Training loss:',loss, ' Test loss:',test_loss)



