import random
import numpy as np
from pydub import AudioSegment


class AudioSegmentGenerator:
    def __init__(self, audio_file, segment_duration_ms, anomaly_probability=0.2, random_sampling=False, max_samples=None, scale_factor=1):
        """
        Initialize the audio segment generator.

        Args:
            audio_file (str): Path to the audio file.
            segment_duration_ms (int): Duration of each audio segment in milliseconds.
            anomaly_probability (float): Probability of injecting an anomaly into a segment.
            random_sampling (bool): Whether to randomly sample segments from the audio file.
            max_samples (int or None): Maximum number of segments to generate (None for unlimited).
            scale_factor (int): Factor by which to scale down the output Y vector. Y will be 1/scale_factor the size of X.
        """
        self.audio = AudioSegment.from_file(audio_file)
        self.segment_duration_ms = segment_duration_ms
        self.anomaly_probability = anomaly_probability
        self.random_sampling = random_sampling
        self.max_samples = max_samples
        self.scale_factor = scale_factor
        self.num_segments = len(self.audio) // self.segment_duration_ms

        if random_sampling:
            self.indices = random.sample(range(self.num_segments), self.num_segments)
        else:
            self.indices = list(range(self.num_segments))

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def _inject_anomaly(self, segment):
        """
        Inject an anomaly into the audio segment.

        Args:
            segment (AudioSegment): The audio segment.

        Returns:
            AudioSegment: The modified audio segment with an anomaly.
        """
        samples = np.array(segment.get_array_of_samples())
        noise = np.random.normal(0, 0.1 * np.max(np.abs(samples)), len(samples)).astype(samples.dtype)
        samples_with_noise = samples + noise
        return segment._spawn(samples_with_noise)

    def _downscale_y(self, segment, label):
        """
        Downscale the Y vector by the scale_factor.

        Args:
            segment (AudioSegment): The audio segment.
            label (int): The anomaly label.

        Returns:
            tuple: A tuple (X, Y) where X is the audio segment and Y is the downscaled label array.
        """
        samples = np.array(segment.get_array_of_samples())
        scaled_length = len(samples) // self.scale_factor
        y_vector = np.full(scaled_length, label, dtype=int)
        return samples, y_vector

    def __len__(self):
        """
        Return the total number of segments available.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a specific segment by index.

        Args:
            idx (int): The index of the segment to retrieve.

        Returns:
            tuple: A tuple (X, Y) where X is the segment data as a NumPy array and Y is the downscaled label array.
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError("Index out of range")

        segment_index = self.indices[idx]
        start_ms = segment_index * self.segment_duration_ms
        end_ms = start_ms + self.segment_duration_ms
        segment = self.audio[start_ms:end_ms]

        if random.random() < self.anomaly_probability:
            segment = self._inject_anomaly(segment)
            label = 1
        else:
            label = 0

        return self._downscale_y(segment, label)

    def __iter__(self):
        """
        Make the generator iterable.
        """
        for idx in range(len(self)):
            yield self[idx]


# Example usage:
if __name__ == "__main__":
    generator = AudioSegmentGenerator(
        "example.wav",
        segment_duration_ms=1000,
        anomaly_probability=0.3,
        random_sampling=True,
        max_samples=10,
        scale_factor=128,
    )

