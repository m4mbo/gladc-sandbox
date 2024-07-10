import matplotlib.pyplot as plt
import numpy as np

class LossTracker:
    def __init__(self):
        self.losses = {
            'reconstruction_loss': [],
            'contrastive_loss': [],
            'node_graph_loss': [],
            'total_loss': []
        }

    def add_losses(self, reconstruction_loss, contrastive_loss, node_graph_loss, total_loss):
        self.losses['reconstruction_loss'].append(reconstruction_loss)
        self.losses['contrastive_loss'].append(contrastive_loss)
        self.losses['node_graph_loss'].append(node_graph_loss)
        self.losses['total_loss'].append(total_loss)

    def plot_losses(self):
        epochs = len(self.losses['reconstruction_loss'][0])
        avg_reconstruction_loss = np.mean(self.losses['reconstruction_loss'], axis=0)
        avg_contrastive_loss = np.mean(self.losses['contrastive_loss'], axis=0)
        avg_node_graph_loss = np.mean(self.losses['node_graph_loss'], axis=0)
        avg_total_loss = np.mean(self.losses['total_loss'], axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), avg_reconstruction_loss, label='Reconstruction Loss')
        plt.plot(range(epochs), avg_contrastive_loss, label='Contrastive Loss')
        plt.plot(range(epochs), avg_node_graph_loss, label='Node + Graph Loss')
        plt.plot(range(epochs), avg_total_loss, label='Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.show()

    def plot_final_losses(self):
        final_reconstruction_losses = [loss[-1] for loss in self.losses['reconstruction_loss']]
        final_contrastive_losses = [loss[-1] for loss in self.losses['contrastive_loss']]
        final_node_graph_losses = [loss[-1] for loss in self.losses['node_graph_loss']]
        final_total_losses = [loss[-1] for loss in self.losses['total_loss']]

        # Calculating averages
        avg_final_reconstruction_loss = np.mean(final_reconstruction_losses)
        avg_final_contrastive_loss = np.mean(final_contrastive_losses)
        avg_final_node_graph_loss = np.mean(final_node_graph_losses)
        avg_final_total_loss = np.mean(final_total_losses)

        # Bar plot setup
        bar_width = 0.2
        r1 = np.arange(4)  # Four types of losses
        r2 = [x + bar_width for x in r1]

        plt.figure(figsize=(10, 6))
        plt.bar(r1, [avg_final_reconstruction_loss, avg_final_contrastive_loss, avg_final_node_graph_loss, avg_final_total_loss],
                color=['b', 'r', 'g', 'y'], width=bar_width, edgecolor='grey', 
                tick_label=['Reconstruction Loss', 'Contrastive Loss', 'Node + Graph Loss', 'Total Loss'])

        plt.xlabel('Loss Type')
        plt.ylabel('Average Loss')
        plt.title('Average Final Losses Across Folds')
        plt.show()