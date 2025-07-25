import numpy      as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
import tensorflow.experimental.numpy as tnp
from tensorflow.keras        import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape
from tensorflow.keras.models import Model


def smape_loss(y_true, y_pred):
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
    :return: Loss value
    """
    # mask=tf.where(y_true,1.,0.)
    mask      = tf.cast(y_true, tf.bool)
    mask      = tf.cast(mask, tf.float32)
    sym_sum   = tf.abs(y_true) + tf.abs(y_pred)
    condition = tf.cast(sym_sum, tf.bool)
    weights   = tf.where(condition, 1. / (sym_sum + 1e-8), 0.0)
    return 200 * tnp.nanmean(tf.abs(y_pred - y_true) * weights * mask)


class NBeatsNet:
    GENERIC_BLOCK     = 'generic'
    TREND_BLOCK       = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'

    def __init__(self,
                 input_dim              = 1,
                 output_dim             = 1,
                 exo_dim                = 0,
                 backcast_length        = 10,
                 forecast_length        = 1,
                 stack_types            = (TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack    = 3,
                 thetas_dim             = (4, 8),
                 share_weights_in_stack = False,
                 hidden_layer_units     = 256,
                 nb_harmonics           = None):

        self.stack_types               = stack_types
        self.nb_blocks_per_stack       = nb_blocks_per_stack
        self.thetas_dim                = thetas_dim
        self.units                     = hidden_layer_units
        self.share_weights_in_stack    = share_weights_in_stack
        self.backcast_length           = backcast_length
        self.forecast_length           = forecast_length
        self.input_dim                 = input_dim
        self.output_dim                = output_dim
        self.exo_dim                   = exo_dim
        self.input_shape               = (self.backcast_length, self.input_dim)
        self.exo_shape                 = (self.backcast_length, self.exo_dim)
        self.output_shape              = (self.forecast_length, self.output_dim)
        self.weights                   = {}
        self.nb_harmonics              = nb_harmonics
        self._gen_intermediate_outputs = False
        self._intermediary_outputs     = []
        assert len(self.stack_types) == len(self.thetas_dim)

        x  = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly    = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k]      = Subtract()([x_[k], backcast[k]])
                    layer_name = f'stack_{stack_id}-{stack_type.title()}Block_{block_id}'
                    if self.input_dim >= 1:
                        layer_name += f'_Dim_{k}'
                    # rename.
                    forecast[k] = Lambda(function=lambda _x: _x, name=layer_name)(forecast[k])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
            x_[k] = Reshape(target_shape=(self.backcast_length, 1))(x_[k])
        if self.input_dim > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.input_dim)])
            x_ = Concatenate()([x_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]
            x_ = x_[0]

        if self.input_dim != self.output_dim:
            y_ = Dense(self.output_dim, activation='linear', name='reg_y')(y_)
            x_ = Dense(self.output_dim, activation='linear', name='reg_x')(x_)

        inputs_x         = [x, e] if self.has_exog() else x
        n_beats_forecast = Model(inputs_x, y_, name=self._FORECAST)
        n_beats_backcast = Model(inputs_x, x_, name=self._BACKCAST)

        self.models    = {model.name: model for model in [n_beats_backcast, n_beats_forecast]}
        self.cast_type = self._FORECAST

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in     a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    @staticmethod
    def name():
        return 'NBeatsKeras'

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):
        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1        = reg(Dense(self.units, activation='relu', name=n('d1')))
        d2        = reg(Dense(self.units, activation='relu', name=n('d2')))
        d3        = reg(Dense(self.units, activation='relu', name=n('d3')))
        d4        = reg(Dense(self.units, activation='relu', name=n('d4')))
        if stack_type == 'generic':
            theta_b  = reg(Dense(nb_poly, 
                                 activation = 'linear', 
                                 use_bias   = False, 
                                 name       = n('theta_b')))
            theta_f  = reg(Dense(nb_poly, 
                                 activation = 'linear', 
                                 use_bias   = False, 
                                 name       = n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_type == 'trend':
            theta_f  = theta_b = reg(Dense(nb_poly, 
                                           activation = 'linear', 
                                           use_bias   = False, 
                                           name       = n('theta_f_b')))
            backcast = Lambda(trend_model, arguments={'is_forecast'     : False, 
                                                      'backcast_length' : self.backcast_length,
                                                      'forecast_length' : self.forecast_length})
            forecast = Lambda(trend_model, arguments={'is_forecast'     : True, 
                                                      'backcast_length' : self.backcast_length,
                                                      'forecast_length' : self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(Dense(self.nb_harmonics, 
                                    activation = 'linear', 
                                    use_bias   = False, 
                                    name       = n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, 
                                    activation = 'linear', 
                                    use_bias   = False, 
                                    name       = n('theta_b')))
            theta_f  = reg(Dense(self.forecast_length, 
                                 activation = 'linear', 
                                 use_bias   = False, 
                                 name       = n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments = {'is_forecast'     : False, 
                                           'backcast_length' : self.backcast_length,
                                           'forecast_length' : self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments = {'is_forecast'     : True, 
                                           'backcast_length' : self.backcast_length,
                                           'forecast_length' : self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_          = d1(d0)
            d2_          = d2(d1_)
            d3_          = d3(d2_)
            d4_          = d4(d3_)
            theta_f_     = theta_f(d4_)
            theta_b_     = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.models[self._FORECAST], name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            cast_type = self._FORECAST
            if attr.__name__ == 'predict' and 'return_backcast' in kwargs and kwargs['return_backcast']:
                del kwargs['return_backcast']
                cast_type = self._BACKCAST

            if attr.__name__ == 'predict' and self._gen_intermediate_outputs:
                import keract
                outputs = keract.get_activations(model=self, x=args)
                self._intermediary_outputs = [
                    {'layer': a, 'value': b} for a, b in outputs.items() if str(a).startswith('stack_')
                ]
            return getattr(self.models[cast_type], attr.__name__)(*args, **kwargs)

        return wrapper


def linear_space(backcast_length, forecast_length, is_forecast=True):
    # ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    # return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))
    horizon = forecast_length if is_forecast else backcast_length
    return K.arange(0, horizon) / horizon


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p      = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t      = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1     = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2     = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))
