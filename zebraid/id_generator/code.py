"""Identity code generation helpers for ZEBRAID.

NOTE: IDs are now assigned by the registry when zebras are added.
This module is deprecated and maintained for backward compatibility only.
"""

from uuid import uuid4


def generate_code(vector=None) -> str:
    """DEPRECATED: Generate a unique ID.
    
    IDs should now be assigned by the registry instead.
    This function is kept for backward compatibility.
    
    Args:
        vector: Optional embedding (ignored)
    
    Returns:
        UUID4 string
    """
    return str(uuid4())


def generate_dual_code(global_vec=None, local_vec=None) -> dict[str, str]:
    """DEPRECATED: Generate dual IDs.
    
    IDs should now be assigned by the registry instead.
    This function is kept for backward compatibility.
    
    Args:
        global_vec: Optional global embedding (ignored)
        local_vec: Optional local embedding (ignored)
    
    Returns:
        Dict with "global" and "local" UUID strings
    """
    return {
        "global": generate_code(),
        "local": generate_code(),
    }

