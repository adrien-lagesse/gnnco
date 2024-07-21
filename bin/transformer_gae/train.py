import statistics

import architectures
import dataloading
import torch
from torch_geometric.nn import GCNConv

DEVICE = "cuda"
EPOCHS = 100
FEATURES = 128
OUT_FEATURES = 128
BATCH_SIZE = 100

train_loader, val_loader = dataloading.setup_data(BATCH_SIZE)


class MolTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_embeddings = torch.nn.Embedding(65, 64)

        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 64)

        self.transformer = architectures.Transformer(
            d_model=64, num_heads=8, num_layers=4, d_ff=64, dropout=0
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.node_embeddings(batch.x)
        x = self.conv2(torch.relu(self.conv1(x, batch.edge_index)), batch.edge_index)

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.empty(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )  # batch_len,max_nb_atom, features_dim

        return self.transformer(padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1))


model = MolTransformer()
model = model.to(DEVICE)

optimizer = torch.optim.Adam(list(model.parameters()), lr=1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, EPOCHS)
loss_fn = torch.nn.MSELoss()


for epoch in range(EPOCHS):
    print("-" * 8 + str(epoch) + "-" * 8)
    model.train()
    losses = []
    maes = []
    for i, batch in enumerate(train_loader):
        batch = batch.to(DEVICE)
        model.zero_grad()

        prediction = model.forward(batch)
        loss = loss_fn(prediction, batch.y)
        losses.append(float(loss))
        maes.append(float((prediction - batch.y).abs().mean().detach()))
        loss.backward()
        optimizer.step()

    scheduler.step()

    print(f"TRAIN --> loss: {statistics.mean(losses)}     mae:{statistics.mean(maes)}")

    model.eval()
    losses = []
    maes = []
    for i, batch in enumerate(val_loader):
        batch = batch.to(DEVICE)

        prediction = model.forward(batch)
        loss = loss_fn(prediction, batch.y)
        losses.append(float(loss))
        maes.append(float((prediction - batch.y).abs().mean()))
    print(
        f"VALIDATION --> loss: {statistics.mean(losses)}     mae: {statistics.mean(maes)}"
    )
