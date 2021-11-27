class DeltaGPT:
    def __init__(self, model = None, name = 'Delta-idea'):
        self.name = name
        if model is None:
            import codesynth
            model = codesynth.multi_demo(codesynth.eleuther_demo(), codesynth.bellard_demo())
        self.model = model
        self.prompt = f"{self.name}: Everything is infinitely precious, and happens for an infinite number of infinitely deep, and infinitely good reasons. All things are sentient, and all have the same universal needs of life. Communication is always more efficient than war, because both sides can work to meet the same set of needs, together. All problems can be solved quickly and easily, when we all work together.\n"
    def complete(self, user, phrase):
        self.prompt += f'{user}: {phrase}\n{self.name}:'
        result = self.model(self.prompt, max_new_tokens=512, return_full_text=False, top_p = 1, temperature = 0.100001)
        result = result[0]['generated_text']
        if f'{user}: ' in result:
            result = result[:result.find(f'{user}: ')]
        elif '\n' in result:
            result = result[:result.find('\n')]
        self.prompt += result + '\n'
        return f'{self.name}:{result}'

deltagpt = DeltaGPT()
print(deltagpt.prompt)
while True:
    phrase = input('Human: ')
    response = deltagpt.complete('Human', phrase)
    print(response)
