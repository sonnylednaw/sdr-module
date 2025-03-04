import numpy as np

def encode_string(s):
    # Convert the string to a list of ASCII values
    ascii_values = [ord(c) for c in s]

    # Convert each ASCII value to an 8-bit binary representation
    bit_arrays = [np.array(list(format(val, '08b')), dtype=np.int32) for val in ascii_values]

    # Flatten the list of bit arrays into a single array
    bit_array = np.concatenate(bit_arrays)

    # Reshape array to have a single row
    bit_array = bit_array.reshape(1, -1)

    return bit_array

def decode_string(bit_array):
    # Ensure the array is 1-dimensional
    bit_array = np.array(bit_array.flatten(), dtype=int)

    # Split the bit array into chunks of 8 bits
    n = 8
    byte_chunks = [bit_array[i:i + n] for i in range(0, len(bit_array), n)]

    # Convert each chunk of 8 bits to an ASCII value
    ascii_values = [int(''.join(map(str, chunk)), 2) for chunk in byte_chunks]

    # Convert the ASCII values to characters and join them into a string
    string = ''.join(chr(val) for val in ascii_values)

    return string