def generar_texto(model, tokenizer, prompt: str, device, max_new_tokens=200, kaggle= False) -> str:
    if kaggle:
        return _generar_texto_kaggle(model, prompt, device, max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # To avoid warning
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    #
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True
    )

    # Cortar exactamente los tokens del prompt
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def _generar_texto_kaggle(model, prompt: str, device, max_new_tokens=200) -> str:
    USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
    MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n"

    # Construir el input en el formato que Gemma-3 espera
    conversation = [[
        USER_CHAT_TEMPLATE.format(prompt=prompt),
        MODEL_CHAT_TEMPLATE
    ]]

    # Generar texto
    output = model.generate(
        conversation,
        device=device,
        output_len=max_new_tokens
    )

    # El modelo devuelve una lista de listas de strings
    # Tomamos la última parte generada
    generated = output[0][-1]

    # Limpiar tokens especiales
    generated = generated.replace("<end_of_turn>", "").strip()

    return generated