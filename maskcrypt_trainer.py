"""The customized trainer designed for MaskCrpyt."""
import logging
import os
import torch
import time
from torch import nn
from typing import OrderedDict
from plato.config import Config
from plato.trainers import basic

class Trainer(basic.Trainer):
    """The trainer with gradient computation when local training is finished."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.gradient = OrderedDict()
        
    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        # Modified the optimizer and scheduler to use Adam and StepLR
        self._loss_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=9e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.8)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, ele in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                # Custom logic for local data
                examples, labels = ele["image"].to(self.device), ele["label"].to(self.device)
                loss = self.perform_forward_and_backward_passes(
                    config, examples, labels
                )
                
                # Log progress
                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)
    
    def test_model(self, config, testset, sampler=None, **kwargs):  # pylint: disable=unused-argument
        """A custom testing loop."""
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config["batch_size"], shuffle=False
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for ele in test_loader:
                # Move examples and labels to the appropriate device
                examples, labels = ele["image"].to(self.device), ele["label"].to(self.device)

                # Forward pass
                outputs = self.model(examples)  # No need to flatten the examples
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy
    
    def train_run_end(self, config):
        """Compute gradients on local data when training is finished."""
        logging.info(
            "[Client #%d] Training completed, computing gradient.", self.client_id
        )
        # Set the existing gradients to zeros
        [x.grad.zero_() for x in list(self.model.parameters())]
        self.model.to(self.device)
        for ele in self.train_loader:
            examples, labels = ele["image"].to(self.device), ele["label"].to(self.device)
            outputs = self.model(examples)
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(outputs, labels)
            loss = loss * (len(labels) / len(self.sampler))
            loss.backward()

        param_dict = dict(list(self.model.named_parameters()))
        state_dict = self.model.state_dict()
        for name in state_dict.keys():
            if name in param_dict:
                self.gradient[name] = param_dict[name].grad
            else:
                self.gradient[name] = torch.zeros(state_dict[name].shape)

        model_type = config["model_name"]
        filename = f"{model_type}_gradient_{self.client_id}_{config['run_id']}.pth"
        self._save_gradient(filename)

    def get_gradient(self):
        """Read gradients from file and return to client."""
        model_type = Config().trainer.model_name
        run_id = Config().params["run_id"]
        filename = f"{model_type}_gradient_{self.client_id}_{run_id}.pth"
        return self._load_gradient(filename)

    def _save_gradient(self, filename=None, location=None):
        """Saving the gradients to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        torch.save(self.gradient, model_path)

    def _load_gradient(self, filename=None, location=None):
        """Load gradients from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"
        return torch.load(model_path, weights_only=False)
