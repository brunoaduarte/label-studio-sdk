from unittest.mock import patch

from label_studio_sdk.converter.exports.yolo import process_keypoints_for_yolo


def test_process_rle_to_yolo(tmp_path):
    labels = [
        {
            'format': 'rle',
            'rle': b'fake',
            'keypointlabels': ['hand'],
            'type': 'KeyPointLabels',
            'original_width': 100,
            'original_height': 100,
            'id': '1550',
            'parentID': None,
        }
    ]
    label_file = tmp_path / 'label.txt'
    cats = [{'id': 0, 'name': 'hand', 'keypoints': ['hand']}]
    mapping = {'hand': 0}

    with patch('label_studio_sdk.converter.exports.yolo.generate_contour_from_rle') as mock_gen:
        mock_gen.return_value = ([[0, 0, 10, 0, 10, 10, 0, 10]], [], [])
        process_keypoints_for_yolo(labels, str(label_file), mapping, cats, False, [])

    with open(label_file) as f:
        content = f.read().strip()

    assert content == '0 0.0 0.0 0.1 0.0 0.1 0.1 0.0 0.1'
