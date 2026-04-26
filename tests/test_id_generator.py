import uuid

import pytest

from zebraid.id_generator import generate_code, generate_dual_code


def test_generate_code_returns_valid_uuid():
    """Test that generate_code returns a valid UUID string.
    
    NOTE: generate_code is deprecated. IDs are now assigned by the registry.
    This test is maintained for backward compatibility.
    """
    code = generate_code()
    
    # Should be a valid UUID string
    uuid_obj = uuid.UUID(code)
    assert str(uuid_obj) == code
    assert len(code) == 36  # UUID format with hyphens


def test_generate_code_is_unique():
    """Test that generate_code produces unique values.
    
    NOTE: generate_code is deprecated. IDs are now assigned by the registry.
    This test is maintained for backward compatibility.
    """
    code1 = generate_code()
    code2 = generate_code()
    
    # UUIDs should be unique
    assert code1 != code2


def test_generate_dual_code_returns_both_codes():
    """Test that generate_dual_code returns both global and local codes.
    
    NOTE: generate_dual_code is deprecated. IDs are now assigned by the registry.
    This test is maintained for backward compatibility.
    """
    result = generate_dual_code()
    
    assert "global" in result
    assert "local" in result
    
    # Both should be valid UUIDs
    uuid.UUID(result["global"])
    uuid.UUID(result["local"])
    
    # They should be different (each a unique UUID)
    assert result["global"] != result["local"]


def test_generate_dual_code_format():
    """Test that dual codes are in UUID format.
    
    NOTE: generate_dual_code is deprecated. IDs are now assigned by the registry.
    This test is maintained for backward compatibility.
    """
    for _ in range(5):
        result = generate_dual_code()
        
        global_code = result["global"]
        local_code = result["local"]
        
        # Both should be 36-character UUID strings
        assert len(global_code) == 36
        assert len(local_code) == 36
        
        # Should be parseable as UUIDs
        uuid.UUID(global_code)
        uuid.UUID(local_code)
