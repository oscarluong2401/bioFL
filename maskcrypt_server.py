"""
A MaskCrypt server with selective homomorphic encryption support.
"""
import torch
import numpy as np
import logging
import os
from plato.servers import fedavg_he
from plato.config import Config


class Server(fedavg_he.Server):
    """A MaskCrypt server with selective homomorphic encryption support."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.last_selected_clients = []
        self.best_accuracy = 0.0
        self.no_improvement_epochs = 0
        self.early_stopping_threshold = 20  # Stop after 10 learning epochs with no significant improvement
        self.min_improvement = 0.001  # Minimum improvement of 0.1%

    def choose_clients(self, clients_pool, clients_count):
        """Choose the same clients every two rounds."""
        if self.current_round % 2 != 0:
            self.last_selected_clients = super().choose_clients(
                clients_pool, clients_count
            )
        return self.last_selected_clients

    def customize_server_payload(self, payload):
        """Customize the server payload before sending to the client."""
        if self.current_round % 2 != 0:
            return self.encrypted_model
        else:
            return self.final_mask

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        if self.current_round % 2 != 0:
            # Clients send mask proposals in odd rounds, conduct mask consensus
            self._mask_consensus(updates)
            return baseline_weights
        else:
            # Clients send model updates in even rounds, conduct aggregation
            aggregated_weights = await super().aggregate_weights(
                updates, baseline_weights, weights_received
            )
            return aggregated_weights

    def _mask_consensus(self, updates):
        """Conduct mask consensus on the reported mask proposals."""
        proposals = [update.payload for update in updates]
        mask_size = len(proposals[0])
        if mask_size == 0:
            self.final_mask = torch.tensor([])
        else:
            interleaved_indices = torch.zeros(
                (sum([len(x) for x in proposals])), dtype=int
            )
            for i in range(len(proposals)):
                interleaved_indices[i :: len(proposals)] = proposals[i]

            _, indices = np.unique(interleaved_indices, return_index=True)
            indices.sort()
            self.final_mask = interleaved_indices[indices]
            self.final_mask = self.final_mask.int().tolist()
    
    async def wrap_up(self) -> None:
        """Wraps up when each round of training is done."""
        self.save_to_checkpoint()

        # Break the loop when the target accuracy is achieved
        target_accuracy = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("[%s] Target accuracy reached.", self)
            await self._close()
        # Early stopping logic
        if self.accuracy - self.best_accuracy > self.min_improvement:
            self.best_accuracy = self.accuracy
            self.no_improvement_epochs = 0
            logging.info(
                f"Testing accuracy improved to {self.accuracy:.4f}. Resetting patience counter."
            )
        else:
            self.no_improvement_epochs += 1
            logging.info(
                f"No significant improvement for {self.no_improvement_epochs} epoch(s)."
            )

        if self.no_improvement_epochs >= self.early_stopping_threshold:
            logging.info(
                f"Stopping training after {self.no_improvement_epochs} epochs with no significant improvement."
            )
            await self._close()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self._close()
    
    async def _close(self):
        """Closes the server."""
        logging.info("[%s] Training concluded.", self)
        self.trainer.save_model("last.pth", "bioFL_trained")

        self.server_will_close()
        self.callback_handler.call_event("on_server_will_close", self)

        await self._close_connections()
        os._exit(0)