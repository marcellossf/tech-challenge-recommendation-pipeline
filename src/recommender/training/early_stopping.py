"""Early stopping desacoplado do loop de treino."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Interrompe treinamento quando a validacao deixa de melhorar."""

    patience: int
    min_delta: float
    best_loss: float = float("inf")
    stale_epochs: int = 0

    def update(self, loss: float) -> bool:
        """Atualiza o estado e informa se o treino deve parar."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.stale_epochs = 0
            return False
        self.stale_epochs += 1
        return self.stale_epochs >= self.patience
