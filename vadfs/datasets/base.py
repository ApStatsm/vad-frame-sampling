"""Dataset identity and evaluation contract, independent of model execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    prompts: tuple
    expected_videos: int
    expected_anomaly_videos: int
    option_module: str
    primary_branch: str
    primary_metric: str


def validate_manifest(manifest, spec):
    videos = manifest['videos']
    if manifest.get('dataset') != spec.key or len(videos) != spec.expected_videos:
        raise ValueError(f'Expected the complete {spec.name} test manifest')
    names = [video['filename'] for video in videos]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate test video names')
    if sum(video['is_anomaly'] for video in videos) != spec.expected_anomaly_videos:
        raise ValueError('Unexpected normal/anomaly split')
    for index, video in enumerate(videos):
        if (video['official_csv_index'] != index or type(video['is_anomaly']) is not bool
                or type(video.get('reference_tokens')) is not int or video['reference_tokens'] < 1
                or video.get('evaluation_frames') != video['reference_tokens'] * 16):
            raise ValueError(f'Invalid evaluation order or timeline for {video["filename"]}')
    return videos
