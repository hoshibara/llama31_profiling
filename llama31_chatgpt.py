import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from torch.profiler import profile, record_function, ProfilerActivity
from functools import partial # Import partial
from datetime import datetime

DEVICE = "xpu"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B" # 实际分析时使用
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # 测试用
USE_COMPILE = True  # Set to True for torch.compile

# MODEL_CONFIGS = {
#     "4-Func": {
#         "hidden_size": 128, 
#         "num_hidden_layers": 4, 
#         "num_attention_heads": 16,
#         "num_key_value_heads": 8, 
#         "intermediate_size": 256, 
#         "max_position_embeddings": 256, 
#         "seq_in": 128
#     },
#     "48-Perf": {
#         "hidden_size": 1024, 
#         "num_hidden_layers": 32, 
#         "num_attention_heads": 32,
#         "num_key_value_heads": 8, 
#         "intermediate_size": 2048, 
#         "max_position_embeddings": 2048, 
#         "seq_in": 1024
#     }
# }
MODEL_CONFIG_NAME = "4-Func-mail"
MODEL_CONFIG = LlamaConfig.from_json_file(f'configs/{MODEL_CONFIG_NAME}.json')
INPUT_ID_LENGTH_DICT = {
    "4-Func": 1024,
    "4-Func-mail": 128,
    "48-Perf-mail": 1024
}


# Input config
PROMPT = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos' Bistro's founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted - it'll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working - I had a few tabs opened on my second monitor all the time. It's actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I'd only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who's conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate's bill. The problem I realized when doing some tests is that people never look at the bill! They think it's some kind of prop, so they never actually read its details. Plus, if you're zoomed out too much, you can't actually read it, so it's hard to know what's going on with the game until you zoom in to the area of a specific plate. One other solution that didn't turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that's indicated by the plate's decoration - its color denotes the team owner. But it's something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious 'Call Waiter' button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team's colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem that I had to face in the end was overscoping. I had too much planned"
MAX_NEW_TOKENS = 5
PROFILE_DIR= f"logs/{MODEL_NAME}_{MODEL_CONFIG_NAME}_COMPILE_{USE_COMPILE}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(PROFILE_DIR, exist_ok=True)

def load_model():
    torch.xpu.empty_cache()
    
    # config_params = MODEL_CONFIGS[MODEL_CONFIG_NAME]
    # Create a LlamaConfig object using the parameters from the chosen configuration
    
    model = AutoModelForCausalLM.from_config(
        MODEL_CONFIG,
    )
    model = model.to(DEVICE)
    return model


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model()
    if USE_COMPILE:
        model.forward = torch.compile(model.forward, dynamic=True)
    model.eval()

    # Tokenize input
    
    inputs = tokenizer(PROMPT, return_tensors="pt", max_length=INPUT_ID_LENGTH_DICT[MODEL_CONFIG_NAME]).to(DEVICE)
    input_ids = inputs.input_ids
    
    schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)
    
    # outputs = model(
    #     input_ids=input_ids,
    #     past_key_values=None,
    #     use_cache=True
    # )

    # === First token ===
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_DIR, worker_name="first_token"),
    ) as prof_first:
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True
        )

    past_key_values = outputs.past_key_values
    print(f"Layer 0 past key shape: {past_key_values[0][0].shape}")  # K of layer 0
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    generated = torch.cat([input_ids, next_token], dim=-1)

    print("\n==== First Token ====")
    print(prof_first.key_averages().table(sort_by="xpu_time_total", row_limit=20))
    print("Generated:", decode(tokenizer, generated[0]))
    first_token_file_path = os.path.join(PROFILE_DIR, "first_token_profile.txt")
    with open(first_token_file_path, "w") as f:
        # Save to file without row limit and with extended column width
        f.write(prof_first.key_averages(group_by_input_shape=True).table(sort_by="xpu_time_total", row_limit=-1, max_name_column_width=100))
    print(f"First token profiling results saved to: {first_token_file_path}")

    # === Rest tokens ===
    for i in range(MAX_NEW_TOKENS - 1):  # already generated one token
        # outputs = model(
        #     input_ids=next_token,
        #     past_key_values=past_key_values,
        #     use_cache=True
        # )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_DIR, worker_name=f"rest_token_{i+1}"),
        ) as prof_step:
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )

        past_key_values = outputs.past_key_values
        print(f"Layer {i+1} past key shape: {past_key_values[0][0].shape}")  # K of layer 0
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

        print(f"\n==== Rest Token {i+1} ====")
        print(prof_step.key_averages().table(sort_by="xpu_time_total", row_limit=20))
        print("Generated:", decode(tokenizer, generated[0]))
        rest_token_file_path = os.path.join(PROFILE_DIR, f"rest_token_{i+1}_profile.txt")
        with open(rest_token_file_path, "w") as f:
            # Save to file without row limit and with extended column width
            f.write(prof_step.key_averages(group_by_input_shape=True).table(sort_by="xpu_time_total", row_limit=-1, max_name_column_width=100))
        print(f"Rest token {i+1} profiling results saved to: {rest_token_file_path}")

    print("\n==== Final Output ====")
    print(decode(tokenizer, generated[0]))

if __name__ == "__main__":
    main()