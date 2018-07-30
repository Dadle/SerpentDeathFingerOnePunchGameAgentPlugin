from skimage.color import rgb2gray
import math
import numpy as np
import matplotlib.pyplot as plt


class ImagePlotter:

    def render_cnn_layer_outputs(self, game_state):
        """
        Plot the cnn layers of the neural network

        :return: None
        """
        idx = np.argmax(game_state.replay_memory.rewards)
        self.plot_state(game_state=game_state, idx=idx)
        self.plot_layer_output(game_state=game_state, layer_name='layer_conv1', state_index=idx, inverse_cmap=False)
        self.plot_layer_output(game_state=game_state, layer_name='layer_conv2', state_index=idx, inverse_cmap=False)
        self.plot_layer_output(game_state=game_state, layer_name='layer_conv3', state_index=idx, inverse_cmap=False)

    @staticmethod
    def plot_images(images, kill_count, health, smooth=True):
        """
        plot both image and motion trace for 9 state images in a 3x3 matrix

        :param images: the state images to plot as numpy arrays
        :param kill_count: The current kill count for the state
        :param health: The current player health for the state
        :param smooth: Defaults to True. If True uses smooting when scaling images
        :return:
        """

        print(images.shape)

        # Create figure with sub-plots.
        fig, axes = plt.subplots(9, 2)

        hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        # Interpolation type.
        if smooth:
            interpolation = 'spline16'  # 'lanczos' # Alternative
        else:
            interpolation = 'nearest'

        for i, ax in enumerate(axes):
            # There may be less than 9 images, ensure it doesn't crash
            if i < len(images):
                # Plot the image and motion-trace for state i.
                # Game frame is index 0 and motion-trace is index 1 in each state
                for k in range(2):
                    # Plot image.
                    # ax.imshow(images[i], interpolation=interpolation)

                    # ax = axes.flat[0]
                    ax[k].imshow(images[i][:, :, k], vmin=0, vmax=255,
                                 interpolation=interpolation, cmap='gray')

                    # Show kill count and health below the image
                    xlabel = "Kill count: {0}\nHealth: {1}".format(kill_count[i], health[i])

                    # Show the classes as the label on the x-axis
                    ax[k].set_xlabel(xlabel)

            # Remove ticks from the plot
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell
        plt.show()

    def plot_layer_output(self, game_state, layer_name, state_index, inverse_cmap=False):
        """
        Plot the output of a convolutional layer.

        :param game_state: A GameState object representing the current state of the running game
        :param layer_name: Name of the convolutional layer.
        :param state_index: Index into the replay-memory for a state that
                            will be input to the Neural Network.
        :param inverse_cmap: Boolean whether to inverse the color-map.
        """

        # Get the given state-array from the replay-memory.
        state = game_state.replay_memory.states[state_index]

        # Get the output tensor for the given layer inside the TensorFlow graph.
        # This is not the value-contents but merely a reference to the tensor.
        layer_tensor = game_state.agent.model.get_layer_tensor(layer_name=layer_name)

        # Get the actual value of the tensor by feeding the state-data
        # to the TensorFlow graph and calculating the value of the tensor.
        values = game_state.agent.model.get_tensor_value(tensor=layer_tensor, state=state)

        # Number of image channels output by the convolutional layer.
        num_images = values.shape[3]

        # Number of grid-cells to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_images))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

        print("Dim. of each image:", values.shape)

        if inverse_cmap:
            cmap = 'gray_r'
        else:
            cmap = 'gray'

        # Plot the outputs of all the channels in the conv-layer.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid image-channels.
            if i < num_images:
                # Get the image for the i'th output channel.
                img = values[0, :, :, i]

                # Plot image.
                ax.imshow(img, interpolation='nearest', cmap=cmap)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_state(self, states):
        """Plot the state in the replay-memory with the given index."""

        # Create figure with a grid of sub-plots.
        num_images = len(states)
        num_grids = math.ceil(math.sqrt(num_images))
        fig, axes = plt.subplots(num_grids, num_grids)

        for i, ax in enumerate(axes.flat):
            # Plot the image from the game-environment.
            if i < num_images:
                ax.imshow(states[i].frame[:, :], vmin=0, vmax=1,
                          interpolation='lanczos', cmap='gray')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()