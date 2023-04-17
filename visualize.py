# This class visualzies the class activation heat maps for an object classificaiton model

# We create a model to get the prediction and the values of activation maps
# We then calculate the gradient of predicted classt with respect to the he last conv layer
# A mean intensity of all the channels is calculated and multiply this with the importance of each neuron
# Then we superimpose this over the original image to see which part contibuted in the specific class prediction


import tensorflow as tf
import numpy as np
import matplotlib.cm as cm

class Visualize:
    def __init__(self,):
        None

    def get_heat_map(self,model,image,last_layer_name,pred_index):
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_layer_name).output, model.output])
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    
        heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()

        return self.plot(image,heatmap)
    
    def plot(self,image,heatmap):
        image = image[0]
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")

        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.4 + image
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        return superimposed_img

    
