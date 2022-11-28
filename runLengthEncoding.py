def run_length_encoding(seq):
    compressed = []
    count = 1
    char = seq[0]
    for i in range(1, len(seq)):
        if seq[i] == char:
            count = count + 1
        else:
            compressed.append([char, count])
            char = seq[i]
            count = 1
    compressed.append([char, count])

    bits = ""
    for i in range(0, len(compressed)):
        if compressed[i][0] != " ":
            for j in compressed[i]:
                bits += str(j)

    return bits


def run_length_decoding(compressed_seq):

    seq = ''
    for i in range(0, len(compressed_seq)):
        if i%2 == 0:
            for j in range(int(compressed_seq[i + 1])):
                seq += compressed_seq[i]

    return (seq)
