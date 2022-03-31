

def _create_examples(lines):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        # if i == 0:
        #     continue
        query= line['words']
        # BIOS
        labels = []
        for x in line['labels']:
            labels.append(x)
        examples.append({"query":query, "labels":labels})
    return examples

def _read_text(input_file):
    lines = []
    with open(input_file,'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words":words,"labels":labels})
                    words = []
                    labels = []
            else:
                splits = line.strip("\n").split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words":words,"labels":labels})
    return lines

def file_reading(path):
    lines = _read_text(path)
    examples = _create_examples(lines)
    return examples


def file_writing(data, path):
    with open(path, "w") as f:
        output = ""
        for example in data:
            words = example['query']
            labels = example.get('labels', None)
            for i, word in enumerate(words):
                word = " " if word == "" else word
                if labels is not None:
                    output += word + " " + labels[i] + "\n"
                else:
                    output += word + "\n"
            output += "\n"
        f.write(output)
                
            


