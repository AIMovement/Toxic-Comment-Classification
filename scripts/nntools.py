class nntools:
    def __init__(self, conf):
        self.lr = conf['LR']
        self.epochs = conf['EPOCHS']
        self.batchsize = conf['BATCHSIZE']
        self.decay = conf['DECAY']
        self.opt = conf['OPTIMIZER']
        self.loss = conf['LOSS']
        self.mets = conf['METRICS']
        self.moment = conf['MOMENTUM']
        self.nesterov = conf['NESTEROV']
        self.indim = conf['INDIM']
        self.random_state = conf['RANDOM_STATE']
        self.early_stopping = conf['EARLY_STOPPING']
        self.model_checkpoint = conf['MODEL_CHECKPOINT']
        self.reduce_lr_on_plateau = conf['REDUCELRONPLATEAU']
        self.regularization = conf['REGULARIZATION']
        self.rnn = conf['RNN']
        self.lookback = conf['LOOKBACK']
        self.ontarget = conf['ONTARGET']

        if not self.ontarget:
            self.x_train = conf['X_train']
            self.y_train = conf['y_train']
            self.x_val = conf['X_val']
            self.y_val = conf['y_val']

            self.reshape_input()

        self.sumflag = True

    def reshape_input(self):
        """
        If recurrent type model is called, the input is reshaped to [batchsize, features, ]
        :return: Reshaped x, y - training + validation
        """
        import numpy as np

        if self.rnn:

            while len(self.x_train) % self.lookback != 0:
                self.x_train = self.x_train[:-1, :]
                self.y_train = self.y_train[:-1]

            self.x_train = np.array([self.x_train]).reshape(int(len(self.x_train)/self.lookback), self.lookback, self.indim)
            self.y_train = self.y_train[::self.lookback]

            while len(self.x_val) % self.lookback != 0:
                self.x_val = self.x_val[:-1, :]
                self.y_val = self.y_val[:-1]

            self.x_val = np.array([self.x_val]).reshape(int(len(self.x_val)/self.lookback), self.lookback, self.indim)
            self.y_val = self.y_val[::self.lookback]


    def mdl_ffnn_v1(self):
        """
        Baseline FFNN dense model v1.
        :return: Keras model object.
        """
        from keras.layers import Input, Dense, BatchNormalization, regularizers
        from keras.models import Model

        input1 = Input(shape=(self.indim,), name='Input')
        x1 = Dense(units=10, activation='relu', kernel_regularizer=regularizers.l1(self.regularization), name='FC-1')(input1)
        x1 = BatchNormalization(momentum=0.99)(x1)
        x1 = Dense(units=10, activation='relu', name='FC-2')(x1)
        x1 = BatchNormalization(momentum=0.99)(x1)
        predicts = Dense(units=1, activation='relu', name='Output')(x1)

        model = Model(inputs=input1, outputs=predicts)

        if self.sumflag:
            model.summary()

        model.compile(optimizer=self.getopt(), loss=self.loss, metrics=[self.mets])

        return model


    def mdl_ffnn_v2(self):
        """
        Baseline FFNN dense model v2.
        :return: Keras model object.
        """
        from keras.layers import Input, Dense, Dropout, BatchNormalization, regularizers
        from keras.models import Model

        input1 = Input(shape=(self.indim,), name='Input')
        x1 = Dense(units=128, activation='relu', kernel_regularizer=regularizers.l1(self.regularization), name='FC-1')(input1)
        #x1 = BatchNormalization()(x1)
        x1 = Dropout(0.2, name='Drop-2')(x1)
        x1 = Dense(units=64, activation='relu', name='FC-2')(x1)
        #x1 = BatchNormalization()(x1)
        x1 = Dense(units=32, activation='relu', name='FC-3')(x1)
        predicts = Dense(units=1, activation='relu', name='Output')(x1)

        model = Model(inputs=input1, outputs=predicts)

        if self.sumflag:
            model.summary()

        model.compile(optimizer=self.getopt(), loss=self.loss, metrics=[self.mets])

        return model


    def mdl_ffnn_v3(self):
        from keras.layers import Input, Dense
        from keras.models import Model

        input1 = Input(shape=(self.indim,), name='Input')
        x1 = Dense(units=10, activation='relu', name='FC-1')(input1)
        x1 = Dense(units=10, activation='relu', name='FC-2')(x1)
        predicts = Dense(units=1, activation='softmax', name='Output')(x1)
        model = Model(inputs=input1, outputs=predicts)

        if self.sumflag:
            model.summary()

        model.compile(optimizer=self.getopt(), loss=self.loss, metrics=[self.mets])

        return model


    def mdl_rnn_v1(self):
        """
        Baseline RNN dense model v1.
        :return: Keras model object.
        """
        from keras.layers import Dense, Input, SimpleRNN, Dropout, regularizers
        from keras.models import Model

        input1 = Input(shape=(self.indim, 1), name='Input')
        x1 = SimpleRNN(units=64, activation='relu', name='RNN-1')(input1)
        FC1 = Dense(units=64, activation='relu', name='FC-1')(x1)
        do1 = Dropout(0.2)(FC1)
        FC2 = Dense(units=32, activation='relu', name='FC-2')(do1)
        predicts = Dense(units=1, activation='relu', name='FC-Output')(FC2)

        model = Model(inputs=input1, outputs=predicts)

        if self.sumflag:
            model.summary()

        model.compile(optimizer=self.getopt(), loss=self.loss, metrics=[self.mets])

        return model


    def mdl_lstm_v1(self):
        """

        :return:
        """
        from keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Recurrent, regularizers
        from keras.models import Model

        input = Input(shape=(self.lookback, self.indim))
        encoded_columns = LSTM(units=8, name='lstm-1')(input)
        dense_1 = Dense(1, activation='relu')(encoded_columns)
        model = Model(inputs=input, outputs=dense_1)
        model.compile(optimizer=self.getopt(), loss=self.loss, metrics=[self.mets])

        if self.sumflag:
            model.summary()

        return model


    def mdl_lstm_v2(self):
        """

        :return:
        """
        from keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Recurrent, regularizers
        from keras.models import Model

        input = Input(shape=(self.lookback, self.indim))
        encoded_columns = LSTM(units=8, kernel_regularizer=regularizers.l1(self.regularization), name='lstm-1')(input)
        dense_1 = Dense(16, activation='relu')(encoded_columns)
        dense_2 = Dense(8, activation='relu')(dense_1)
        dense_3 = Dense(1, activation='relu')(dense_2)
        model = Model(inputs=input, outputs=dense_3)
        model.compile(optimizer=self.getopt(), loss=self.loss, metrics=[self.mets])

        if self.sumflag:
            model.summary()

        return model


    def train(self, model):
        """
        Train Keras model.
        :param model: Keras model object.
        :return: Keras history object.
        """

        callback = self.getcallback()
        mdlhist = model.fit(x=self.x_train,
                            y=self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            batch_size=self.batchsize,
                            epochs=self.epochs,
                            callbacks=callback)


        return mdlhist


    def predict(self, model, dat):
        """
        Predict output from Keras model.
        :param model: Keras model object.
        :param dat: Data to predict on.
        :return: Numpy array with predictions.
        """

        model.load_weights(filepath='.mdl_wts.hdf5')

        preds = model.predict(x=dat,
                              batch_size=self.batchsize,
                              verbose=1)

        return preds


    def getcallback(self):
        """
        Get callback
        :return:
        """
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

        if self.early_stopping:
            earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, min_delta=0.001, mode='auto')

        if self.model_checkpoint:
            mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')

        if self.reduce_lr_on_plateau:
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4)

        return [earlyStopping, mcp_save, reduce_lr_loss]


    def getopt(self):
        """
        Get Keras optimizer object from specified optimizer within the object CONF.
        :return: Keras optimizer object.
        """
        import sys
        from keras import optimizers

        if self.opt.upper() == 'SGD':
            opt = optimizers.SGD(lr=self.lr, decay=self.decay, momentum=self.moment, nesterov=self.nesterov)

        elif self.opt.upper() == 'ADAM':
            opt = optimizers.Adam(lr=self.lr, decay=self.decay)

        elif self.opt.upper() == 'RMSPROP':
            opt = optimizers.RMSprop(lr=self.lr, decay=self.decay)

        else:
            sys.exit()

        return opt
