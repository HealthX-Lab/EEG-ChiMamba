import time
import numpy as np
import torch



class EegRandomCrop(object):
    """Randomly crop the EEG data to a given size.

    Args:
        crop_length (int): Desired output signal length.
        length_limit (int, optional): Signal length limit to use.
        multiple (int, optional): Desired number of cropping.
        latency (int, optional): Latency signal length to exclude after record starting.
        return_timing (bool, optional): Decide whether return the sample timing or not.
    """
    def __init__(self, crop_length: int, length_limit: int = 10**7,
                 multiple: int = 1, latency: int = 0, segment_simulation=False, return_timing: bool = False, bands=False):
        if isinstance(crop_length, int) is False:
            raise ValueError(f'{self.__class__.__name__}.__init__(crop_length) '
                             f'needs a integer to initialize')
        if isinstance(crop_length, int) is False or multiple < 1:
            raise ValueError(f'{self.__class__.__name__}.__init__(multiple)'
                             f' needs a positive integer to initialize')
        if isinstance(latency, int) is False or latency < 0:
            raise ValueError(f'{self.__class__.__name__}.__init__(latency)'
                             f' needs a non negative integer to initialize')

        self.crop_length = crop_length
        self.length_limit = length_limit
        self.multiple = multiple
        self.latency = latency
        self.segment_simulation = segment_simulation
        self.return_timing = return_timing
        self.bands=bands

    def __call__(self, sample):
        signal = sample['signal']
        signal_length = min(signal.shape[-1], self.length_limit)

        if self.multiple == 1:
            ct = np.random.randint(self.latency, signal_length - self.crop_length)
            if self.segment_simulation:
                ct = int((ct - self.latency) / self.crop_length) * self.crop_length + self.latency
            
            if self.bands:
                sample['signal'] = signal[:, :, ct:ct + self.crop_length] # [Channels, Bands, Timesteps]
            else:
                sample['signal'] = signal[:, ct:ct + self.crop_length]
            if self.return_timing:
                sample['crop_timing'] = ct
        else:
            signals = []
            crop_timings = []

            for r in range(self.multiple):
                ct = np.random.randint(self.latency, signal_length - self.crop_length)
                if self.segment_simulation:
                    ct = int((ct - self.latency) / self.crop_length) * self.crop_length + self.latency
                
                if self.bands:
                    signals.append(signal[:, :, ct:ct + self.crop_length]) # [Channels, Bands, Timesteps]
                else:
                    signals.append(signal[:, ct:ct + self.crop_length])
                
                crop_timings.append(ct)

            sample['signal'] = signals
            if self.return_timing:
                sample['crop_timing'] = crop_timings

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_length={self.crop_length}, length_limit={self.length_limit}, " \
               f"multiple={self.multiple}, latency={self.latency}, return_timing={self.return_timing})"



class EegDropChannels(object):
    """Drop the specified channel from EEG signal.

    Args:
        index (int or list): Channel index(or induce) to drop.
    """
    def __init__(self, index):
        self.drop_index = index

    def drop_specific_channel(self, signal):
        return np.delete(signal, self.drop_index, axis=0)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self.drop_specific_channel(s))
            sample['signal'] = signals
        else:
            sample['signal'] = self.drop_specific_channel(signal)

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(drop_index={self.drop_index})"


class EegToTensor(object):
    """Convert EEG numpy array in sample to Tensors."""

    @staticmethod
    def _signal_to_tensor(signal):
        if isinstance(signal, (np.core.memmap,)):
            return torch.tensor(signal).to(dtype=torch.float32)
        return torch.from_numpy(signal).to(dtype=torch.float32)

    def __call__(self, sample):
        signal = sample['signal']

        if isinstance(signal, (np.ndarray,)):
            sample['signal'] = self._signal_to_tensor(signal)
        elif isinstance(signal, (list,)):
            signals = []
            for s in signal:
                signals.append(self._signal_to_tensor(s))
            sample['signal'] = signals
        else:
            raise ValueError(f'{self.__class__.__name__}.__call__(sample["signal"]) needs to be set to np.ndarray '
                             f'or their list')

        sample['age'] = torch.tensor(sample['age'], dtype=torch.float32)
        if 'class_label' in sample.keys():
            sample['class_label'] = torch.tensor(sample['class_label'])

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def eeg_collate_fn(batch):
    batched_sample = {k: [] for k in batch[0].keys()}

    for sample in batch:
        if isinstance(sample['signal'], (np.ndarray,)) or torch.is_tensor(sample['signal']):
            for k in sample.keys():
                batched_sample[k] += [sample[k]]

        elif isinstance(sample['signal'], (list,)):
            multiple = len(sample['signal'])

            for s in sample['signal']:
                batched_sample['signal'] += [s]

            for k in sample.keys():
                if k not in ['signal', 'crop_timing']:
                    batched_sample[k] += multiple * [sample[k]]
                elif k == 'crop_timing':
                    batched_sample[k] += [*sample[k]]

    batched_sample['signal'] = torch.stack(batched_sample['signal'])
    batched_sample['age'] = torch.stack(batched_sample['age'])
    if 'class_label' in batched_sample.keys():
        batched_sample['class_label'] = torch.stack(batched_sample['class_label'])

    return batched_sample


class EegToDevice(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        device: Desired working device.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, sample):
        sample['signal'] = sample['signal'].to(self.device)
        sample['age'] = sample['age'].to(self.device)
        sample_keys = sample.keys()
        if 'class_label' in sample_keys:
            sample['class_label'] = sample['class_label'].to(self.device)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"


class EegNormalizePerSignal(torch.nn.Module):
    """Normalize multichannel EEG signal by its internal statistics."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, sample):
        signal = sample['signal']
        std, mean = torch.std_mean(signal, dim=-1, keepdim=True)
        signal.sub_(mean).div_(std + self.eps)
        sample['signal'] = signal

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"

class EegNormalizeMinMax(torch.nn.Module):
    """Scale data to lie between 0 and 1."""

    def __init__(self):
        super().__init__()


    def forward(self, sample):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

class EegNormalizeMeanStd(torch.nn.Module):
    """Normalize multichannel EEG signal by pre-calculated statistics."""

    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        if isinstance(mean, np.ndarray):
            self.mean = torch.from_numpy(mean)
        elif isinstance(mean, list):
            self.mean = torch.tensor(mean)
        elif torch.is_tensor(mean):
            self.mean = mean
        else:
            raise ValueError(f'{self.__class__.__name__}.__init__(mean) needs to be set to among of torch.tensor, '
                             f'np.ndarray, or list')

        if isinstance(std, np.ndarray):
            self.std = torch.from_numpy(std)
        elif isinstance(std, list):
            self.std = torch.tensor(std)
        elif torch.is_tensor(std):
            self.std = std
        else:
            raise ValueError(f'{self.__class__.__name__}.__init__(std) needs to be set to among of torch.tensor, '
                             f'np.ndarray, or list')
        self.eps = eps
        self.std_eps = self.std + self.eps

    def forward(self, sample):
        signal = sample['signal']

        if self.mean.get_device() != signal.get_device():
            self.mean = torch.as_tensor(self.mean, device=signal.device)
            self.std_eps = torch.as_tensor(self.std_eps, device=signal.device)

        signal.sub_(self.mean).div_(self.std_eps)
        sample['signal'] = signal

        if 'target' in sample.keys(): 
            target = sample['target']
            target.sub_(self.mean).div_(self.std_eps)
            sample['target'] = target

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean.squeeze()},std={self.std.squeeze()},eps={self.eps})"


class EegAdditiveGaussianNoise(torch.nn.Module):
    """Additive white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        signal = sample['signal']
        noise = torch.normal(mean=torch.ones_like(signal) * self.mean,
                             std=torch.ones_like(signal) * self.std)
        sample['signal'] = signal + noise
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean},std={self.std})"


class EegMultiplicativeGaussianNoise(torch.nn.Module):
    """Multiplicative white Gaussian noise."""

    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        signal = sample['signal']
        noise = torch.normal(mean=torch.ones_like(signal) * self.mean,
                             std=torch.ones_like(signal) * self.std)
        sample['signal'] = signal + (signal * noise)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean},std={self.std})"


class EegNormalizeAge(torch.nn.Module):
    """Normalize age of EEG metadata by the calculated statistics.

    Args:
        mean: Mean age of all people in EEG training dataset.
        std: Standard deviation of the age for all people in EEG training dataset.
        eps: Small number to prevent zero division.
    """

    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
        self.std_eps = self.std + self.eps

    def forward(self, sample):
        age = sample['age']

        if not torch.is_tensor(self.mean) or self.mean.get_device() != age.get_device():
            self.mean = torch.as_tensor(self.mean, device=age.device)
            self.std_eps = torch.as_tensor(self.std_eps, device=age.device)

        age.sub_(self.mean).div_(self.std_eps)
        sample['age'] = age
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean},std={self.std},eps={self.eps})"


class EegAddGaussianNoiseAge(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        mean: Desired mean of noise level for the age value.
        std: Desired standard deviation of noise level for the age value.
    """

    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        age = sample['age']
        noise = torch.normal(mean=torch.ones_like(age) * self.mean,
                             std=torch.ones_like(age) * self.std)
        sample['age'] = age + noise
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean},std={self.std})"


class EegAgeBias(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        bias: Desired bias to add on age value.
    """

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def forward(self, sample):
        sample['age'] += self.bias
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bias={self.bias})"


class EegAgeZero(torch.nn.Module):
    """Add a Gaussian noise to the age value

    Args:
        bias: Desired bias to add on age value.
    """

    def __init__(self, bias=0.0):
        self.bias = bias
        super().__init__()

    def forward(self, sample):
        sample['age'] = torch.zeros_like(sample['age']) + self.bias
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bias={self.bias})"


class EegSpectrogram(torch.nn.Module):
    """Transform the multichannel 1D sequence as multichannel 2D image using short-time fourier transform
    (a.k.a. Spectrogram) """

    def __init__(self, n_fft, complex_mode='as_real', **kwargs):
        super().__init__()
        if complex_mode not in ('as_real', 'power', 'remove'):
            raise ValueError('complex_mode must be set to one of ("as_real", "power", "remove")')

        self.n_fft = n_fft
        self.complex_mode = complex_mode
        self.stft_kwargs = kwargs

    def _spectrogram(self, x):
        if len(x.shape) == 3:
            N = x.shape[0]

            for i in range(N):
                xf = torch.stft(x[i], n_fft=self.n_fft, return_complex=True, **self.stft_kwargs)

                if i == 0:
                    if self.complex_mode == 'as_real':
                        x_out = torch.zeros((N, 2 * xf.shape[0], xf.shape[1], xf.shape[2]),
                                            dtype=x.dtype, device=x.device)
                    else:
                        x_out = torch.zeros((N, *xf.shape),
                                            dtype=x.dtype, device=x.device)

                if self.complex_mode == 'as_real':
                    x_out[i] = torch.cat((torch.view_as_real(xf)[..., 0],
                                          torch.view_as_real(xf)[..., 1]), dim=0)
                elif self.complex_mode == 'power':
                    x_out[i] = xf.abs()
                elif self.complex_mode == 'remove':
                    x_out[i] = torch.real(xf)

        elif len(x.shape) == 2:
            xf = torch.stft(x, n_fft=self.n_fft, return_complex=True)

            if self.complex_mode == 'as_real':
                x_out = torch.cat((torch.view_as_real(xf)[..., 0],
                                      torch.view_as_real(xf)[..., 1]), dim=0)
            elif self.complex_mode == 'power':
                x_out = xf.abs()
            elif self.complex_mode == 'remove':
                x_out = torch.real(xf)

        else:
            raise ValueError(f'{self.__class__.__name__}._spectrogram(sample["signal"]) '
                             f'- check the signal tensor size.')

        return x_out

    def forward(self, sample):
        signal = sample['signal']
        if torch.is_tensor(signal) is False:
            raise TypeError('Before transforming the data signal as a spectrogram '
                            'it must be converted to a PyTorch Tensor object using EegToTensor() transform.')

        sample['signal'] = self._spectrogram(signal)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_fft={self.n_fft}, complex_mode={self.complex_mode}, " \
               f"stft_kwargs={self.stft_kwargs})"


class TransformTimeChecker(object):
    def __init__(self, instance, header='', str_format=''):
        self.instance = instance
        self.header = header
        self.str_format = str_format

    def __call__(self, sample):
        start = time.time()
        sample = self.instance(sample)
        end = time.time()
        print(f'{self.header + type(self.instance).__name__:{self.str_format}}> {end - start :.5f}')
        return sample


def trim_trailing_zeros(a):
    assert type(a) == np.ndarray
    trim = 0
    for i in range(a.shape[-1]):
        if np.any(a[..., -1 - i] != 0):
            trim = i
            break
    a = a[..., :-trim]
    return a
