import evaluate

if __name__ == "__main__":
    rouge = evaluate.load("rouge")
    predictions = ["hello there", "general kenobi"]
    references = [["hello", "there"], ["general kenobi", "general yoda"]]
    results = rouge.compute(predictions=predictions, references=references)
    print(results)
    print(results["rougeL"])
