This model is created for educational purposes/providing training on transformers. Feel free to use it for your educational purposes as well.

The encoder/decoder implementation is based on the amazing https://www.youtube.com/watch?v=ISNdQcPhsts&t=6268s and the equally amazing Chapter 3 of ISBN-10 1098136799, and I basically have merged both approaches.

If you put the model in 'edu' mode, you can nicely follow the matrix computations during training and inferencing. I'm providing a mini local training set "en-fr.tiny" which should be used in 'edu' mode. Obviously the model won't be trained well with it, but the intent here is to let the student follow the flow of the data through the model and to show how the inputs/outputs are processed.
