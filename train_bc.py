import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import BCPolicy


def fit_bc(states, actions, epochs=100, batch_size=64, lr=0.001,
           device=None, verbose=True, loss_history=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-7

    states_norm = (states - state_mean) / state_std
    dataset = TensorDataset(torch.from_numpy(states_norm), torch.from_numpy(actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCPolicy(state_dim=states.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    iterator = tqdm(range(epochs), desc="BC training") if verbose else range(epochs)
    for epoch in iterator:
        epoch_loss = 0.0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            pred = model(batch_states)
            loss = loss_fn(pred, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if loss_history is not None:
            loss_history.append(avg_loss)
        if verbose:
            iterator.set_postfix(loss=f"{avg_loss:.6f}")
            if (epoch + 1) % 10 == 0:
                print(f"\n[epoch {epoch+1}/{epochs}] loss={avg_loss:.6f}", flush=True)

    return model, state_mean.astype(np.float32), state_std.astype(np.float32)


def _load_transitions(demos_path):
    demos = np.load(demos_path, allow_pickle=True)
    states, actions = [], []
    for traj in demos:
        for step in traj:
            states.append(step["state"])
            actions.append(step["action"])
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def train_bc(demos_path="demos.npy", epochs=100, batch_size=64, lr=0.001, save_path="bc_policy.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    states, actions = _load_transitions(demos_path)
    print(f"Loaded {len(states)} transitions")

    losses = []
    model, state_mean, state_std = fit_bc(states, actions, epochs=epochs, batch_size=batch_size, lr=lr, device=device, loss_history=losses)

    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
    }, save_path)
    print(f"Saved BC policy to {save_path}")

    return losses


if __name__ == "__main__":
    losses = train_bc()
    print(f"Final loss: {losses[-1]:.6f}")
