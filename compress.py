"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

# import time

from huffman import HuffmanTree
from utils import *


# import cProfile


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for item in text:
        if item not in d:
            d[item] = 0
        d[item] += 1
    return d


def huffman_sorting_dict_helper(freq_dict: dict[int, int]) -> \
        tuple[dict[int, int], list[int]]:
    """
    sort the freq_dict according to the ascending order of values
    """
    sorted_dict = {}
    sorted_keys = sorted(freq_dict, key=freq_dict.get)
    for w in sorted_keys:
        sorted_dict[w] = freq_dict[w]
    return sorted_dict, sorted_keys


def huffman_subtrees_helper(s: list[int], all_huff_trees: list[HuffmanTree],
                            sd: dict[int, int], name: str) -> HuffmanTree:
    """
    return a newly created huffman tree of two lowest frequency
    """
    if len(s) >= 2:
        s1 = s[0]
        s2 = s[1]
        sd[name] = sd[s1] + sd[s2]
        if isinstance(s1, str) and isinstance(s2, str) \
                and s1[:9] == 'nonsymbol' and s2[:9] == 'nonsymbol':
            store = HuffmanTree(None, all_huff_trees[0], all_huff_trees[1])
            all_huff_trees.pop(0)
            all_huff_trees.pop(0)
            return store
        if isinstance(s1, str) and s1[:9] == 'nonsymbol':
            store = all_huff_trees[0]
            all_huff_trees.pop(0)
            return HuffmanTree(None, store, HuffmanTree(s[1]))
        if isinstance(s2, str) and s2[:9] == 'nonsymbol':
            store = all_huff_trees[0]
            all_huff_trees.pop(0)
            return HuffmanTree(None, HuffmanTree(s[0]), store)
        else:
            return HuffmanTree(None, HuffmanTree(s[0]), HuffmanTree(s[1]))
    return None


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if freq_dict == {}:
        return HuffmanTree(None, None, None)

    all_huffman_trees = []
    all_sorted_dict = {}

    i = 1
    if len(freq_dict) == 1:
        key = 0
        for k in freq_dict:
            key = k
        freq_dict[(key + 1) % 256] = 0

    while len(freq_dict) > 1:
        sorted_dict, sorted_keys = huffman_sorting_dict_helper(freq_dict)
        all_sorted_dict[i] = sorted_dict.copy()
        i += 1
        all_huffman_trees.append(
            huffman_subtrees_helper(sorted_keys, all_huffman_trees,
                                    sorted_dict,
                                    f'nonsymbol{i}'))
        sorted_dict.pop(sorted_keys[0])
        sorted_dict.pop(sorted_keys[1])
        freq_dict = sorted_dict
    return all_huffman_trees[0]


def helper_get_codes(tree: HuffmanTree, code_dict: dict[int, int], c: str,
                     index: int) -> dict[int, str]:
    """
    return a dictionary of codes
    """
    if tree.is_leaf():
        code_dict[tree.symbol] = c
    else:
        c += "0"
        index += 1
        helper_get_codes(tree.left, code_dict, c, index)
        c = c[:len(c) - 1]
        c += "1"
        helper_get_codes(tree.right, code_dict, c, index)
        c = c[:len(c) - 1]
    return code_dict


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return helper_get_codes(tree, {}, '', 0)


def helper_number_nodes(tree: HuffmanTree) -> None:
    """
    number the nodes
    """
    if tree.is_leaf():
        return None
    else:
        helper_number_nodes(tree.left)
        helper_number_nodes(tree.right)
        tree.number = helper_number_nodes.give_num
        helper_number_nodes.give_num += 1
    return None


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    helper_number_nodes.give_num = 0
    helper_number_nodes(tree)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    if freq_dict == {}:
        return 0
    add, numbers = 0, 0
    for key, value in find_depth(tree, {}, 0).items():
        add += freq_dict[key] * value
        numbers += freq_dict[key]
    return add / numbers


def find_depth(t: HuffmanTree, depth_dict: dict[int, int],
               length: int) -> dict[int, int]:
    """
    return the dictionary containing depth/height of leaves
    """
    if t.is_leaf():
        if t.symbol not in depth_dict:
            depth_dict[t.symbol] = 0
        depth_dict[t.symbol] += length
    else:
        length += 1
        find_depth(t.left, depth_dict, length)
        find_depth(t.right, depth_dict, length)
        length -= 1
    return depth_dict


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    byte_lst, compress_string = [], ''
    for i in text:
        for k in codes[i]:
            compress_string += k
            if len(compress_string) % 8 == 0:
                byte_lst.extend([compress_string])
                compress_string = ''
    if compress_string != '':
        byte_lst.extend([compress_string])
    return bytes([bits_to_byte(j) for j in byte_lst])


def helper_trees_to_bytes(trees_to_bytes_lst: list, tree: HuffmanTree) -> bytes:
    """
    Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.
    """
    if tree.is_leaf():
        return bytes([])
    else:
        helper_trees_to_bytes(trees_to_bytes_lst, tree.left)
        helper_trees_to_bytes(trees_to_bytes_lst, tree.right)
        if tree.left.is_leaf():
            trees_to_bytes_lst.extend([0, tree.left.symbol])
        if not tree.left.is_leaf():
            trees_to_bytes_lst.extend([1, tree.left.number])
        if tree.right.is_leaf():
            trees_to_bytes_lst.extend([0, tree.right.symbol])
        if not tree.right.is_leaf():
            trees_to_bytes_lst.extend([1, tree.right.number])
    return bytes(trees_to_bytes_lst)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    return helper_trees_to_bytes([], tree)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None),\
    HuffmanTree(12, None, None)), HuffmanTree(None,\
    HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    if root_index >= len(node_lst):
        return None
    all_huffman_trees, index = {}, 0
    for i in range(root_index + 1):
        if node_lst[i].l_type == 0 and node_lst[i].r_type == 0:
            all_huffman_trees[i] = HuffmanTree(None,
                                               HuffmanTree(node_lst[i].l_data),
                                               HuffmanTree(node_lst[i].r_data))
        if node_lst[i].l_type == 1 and node_lst[i].r_type == 1:
            all_huffman_trees[i] = HuffmanTree(None, all_huffman_trees[
                node_lst[i].l_data], all_huffman_trees[node_lst[i].r_data])
        if node_lst[i].l_type == 1 and node_lst[i].r_type == 0:
            all_huffman_trees[i] = HuffmanTree(None, all_huffman_trees[
                node_lst[i].l_data], HuffmanTree(node_lst[i].r_data))
        if node_lst[i].l_type == 0 and node_lst[i].r_type == 1:
            all_huffman_trees[i] = HuffmanTree(None,
                                               HuffmanTree(node_lst[i].l_data),
                                               all_huffman_trees[
                                                   node_lst[i].r_data])
        index += 1
    return all_huffman_trees[index - 1]


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None),
    HuffmanTree(7, None, None)), HuffmanTree(None,
    HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    if root_index >= len(node_lst):
        return None
    all_huffman_trees, index = {}, 0
    for i in range(root_index + 1):
        if node_lst[i].l_type == 0 and node_lst[i].r_type == 0:
            all_huffman_trees[i] = HuffmanTree(None,
                                               HuffmanTree(node_lst[i].l_data),
                                               HuffmanTree(node_lst[i].r_data))
        if node_lst[i].l_type == 1 and node_lst[i].r_type == 1:
            all_huffman_trees[i] = HuffmanTree(None, all_huffman_trees[0],
                                               all_huffman_trees[1])
        if node_lst[i].l_type == 1 and node_lst[i].r_type == 0:
            all_huffman_trees[i] = HuffmanTree(None, all_huffman_trees[0],
                                               HuffmanTree(node_lst[i].r_data))
        if node_lst[i].l_type == 0 and node_lst[i].r_type == 1:
            all_huffman_trees[i] = HuffmanTree(None,
                                               HuffmanTree(node_lst[i].l_data),
                                               all_huffman_trees[1])
        index += 1
    return all_huffman_trees[index - 1]


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    one_string = ''.join([byte_to_bits(r) for r in list(text)])
    swap_code_dict = dict((v, u) for u, v in get_codes(tree).items())
    store, code = [], ''
    for i in one_string:
        code += i
        if code in swap_code_dict:
            store.append(swap_code_dict[code])
            code = ''
    return bytes(store[:size])


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """

    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
    # import python_ta
    #
    # python_ta.check_all(config={
    #     'allowed-io': ['compress_file', 'decompress_file'],
    #     'allowed-import-modules': [
    #         'python_ta', 'doctest', 'typing', '__future__',
    #         'time', 'utils', 'huffman', 'random'
    #     ],
    #     'disable': ['W0401']
    # })
    #
    # mode = input(
    #     "Press c to compress, d to decompress, or other key to exit: ")
    # if mode == "c":
    #     fname = input("File to compress: ")
    #     start = time.time()
    #     cProfile.run('compress_file(fname, fname + ".huf")')
    #     print(f"Compressed {fname} in {time.time() - start} seconds.")
    # elif mode == "d":
    #     fname = input("File to decompress: ")
    #     start = time.time()
    #     cProfile.run('decompress_file(fname, fname + ".orig")')
    #     print(f"Decompressed {fname} in {time.time() - start} seconds.")
