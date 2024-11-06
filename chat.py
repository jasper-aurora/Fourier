import numpy as np
import matplotlib.colors as mcolors

def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def color_to_rgb(color):
    """Convert color name or hex string to RGB tuple."""
    if color.startswith('#'):
        return hex_to_rgb(color)
    else:
        # Convert named colors to RGB using matplotlib
        rgb = mcolors.to_rgb(color)
        return rgb

def rgb_to_hex(rgb_color):
    """Convert RGB tuple to hex color string."""
    return '#' + ''.join(f'{int(c * 255):02x}' for c in rgb_color)

def interpolate_color(value, start_value, stop_value, color_list):
    """
    Interpolates a color based on a single value or an array of values within a specified range of colors.
    
    Parameters:
    - value: A scalar or numpy array of input values (should be between start_value and stop_value).
    - start_value: The starting value of the range.
    - stop_value: The ending value of the range.
    - color_list: A list of colors in hex format or standard color names (e.g., ['red', '#00FF00', '#0000FF']).
    
    Returns:
    - Hex string(s) of the interpolated color(s).
    """
    # Convert value to a NumPy array if it isn't already
    value = np.asarray(value, dtype=np.float64)

    # Handle the case where value is a scalar
    if value.ndim == 0:
        value = np.expand_dims(value, axis=0)

    num_colors = len(color_list)

    # Normalize the values to [0, 1]
    t = (value - start_value) / (stop_value - start_value)
    t = np.clip(t, 0, 1)  # Ensure t is between 0 and 1

    # Scale t to the range of color indices
    t_scaled = t * (num_colors - 1)
    
    # Determine the lower and upper indices
    lower_indices = np.floor(t_scaled).astype(int)
    upper_indices = np.ceil(t_scaled).astype(int)

    # Ensure upper indices are within bounds
    upper_indices = np.clip(upper_indices, 0, num_colors - 1)

    # Calculate the fractional part for interpolation
    fractional_t = t_scaled - lower_indices
    
    # Prepare an array to hold the interpolated RGB colors
    interpolated_rgb = np.zeros((*t.shape, 3))

    # Interpolate colors
    for i in range(len(value)):
        lower_color = np.array(color_to_rgb(color_list[lower_indices[i]]))
        upper_color = np.array(color_to_rgb(color_list[upper_indices[i]]))
        
        interpolated_rgb[i] = (1 - fractional_t[i]) * lower_color + fractional_t[i] * upper_color

    # Convert interpolated RGB back to hex
    return np.array([rgb_to_hex(color) for color in interpolated_rgb])

# Example usage
start_num = 0
stop_num = 100
color_list = ['yellow', '#00FF00', '#0000FF']  # Standard color name, hex, and more

# Test with a scalar value
value_scalar = 50
result_color_scalar = interpolate_color(value_scalar, start_num, stop_num, color_list)
print(f'Interpolated color for scalar value {value_scalar}: {result_color_scalar}')

# Test with an array of values
value_array = np.array([0, 25, 50, 75, 100])
result_color_array = interpolate_color(value_array, start_num, stop_num, color_list)
print(f'Interpolated colors for array values {value_array}: {result_color_array}')
