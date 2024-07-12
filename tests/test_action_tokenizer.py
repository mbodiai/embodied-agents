import pytest
from mbodied.agents.motion.rt_pali.action_tokenizer.action_tokenizer import ActionTokenizer

@pytest.fixture
def action_tokenizer():
    return ActionTokenizer()

@pytest.fixture
def pose_data():
    return {
        'x': 0.2,
        'y': 0.5,
        'z': 0.8,
        'roll': 0.33,
        'pitch': 0.67,
        'yaw': 0.95,
        'grasp': 1.0
    }

def test_discretize_values(action_tokenizer, pose_data):
    discrete_data = action_tokenizer.discretize_values(pose_data)

    assert discrete_data['x'] == 'ra_51'   
    assert discrete_data['y'] == 'ra_127'  
    assert discrete_data['z'] == 'ra_204' 
    assert discrete_data['roll'] == 'ra_84'  
    assert discrete_data['pitch'] == 'ra_171'  
    assert discrete_data['yaw'] == 'ra_242' 
    assert discrete_data['grasp'] == 'ra_255'

def test_reverse_discretize_values(action_tokenizer, pose_data):
    discrete_data = action_tokenizer.discretize_values(pose_data)
    inverse_data = action_tokenizer.reverse_discretize_values(discrete_data)

    assert pytest.approx(inverse_data['x'], rel=1e-2) == 0.2, f"Value mismatch for x: {inverse_data['x']}"
    assert pytest.approx(inverse_data['y'], rel=1e-2) == 0.5, f"Value mismatch for y: {inverse_data['y']}"
    assert pytest.approx(inverse_data['z'], rel=1e-2) == 0.8, f"Value mismatch for z: {inverse_data['z']}"
    assert pytest.approx(inverse_data['roll'], rel=1e-2) == 0.33, f"Value mismatch for roll: {inverse_data['roll']}"
    assert pytest.approx(inverse_data['pitch'], rel=1e-2) == 0.67, f"Value mismatch for pitch: {inverse_data['pitch']}"
    assert pytest.approx(inverse_data['yaw'], rel=1e-2) == 0.95, f"Value mismatch for yaw: {inverse_data['yaw']}"
    assert pytest.approx(inverse_data['grasp'], rel=1e-2) == 1.0, f"Value mismatch for grasp: {inverse_data['grasp']}"

if __name__ == "__main__":
    pytest.main()
    