from transformer_hw import *
import unittest
    
class TestTransformerFunctions(unittest.TestCase):
    def rand_float_test(self, cls, shape):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        random_input = torch.randn(shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        self.assertEqual(random_input.shape, output.shape)

    def rand_int_test(self, cls, shape):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        random_input = torch.randint(100, 1000, shape).to(device)

        output = layer(random_input)
        if isinstance(output, tuple):
                output = output[0]
        self.assertEqual(random_input.shape, output.shape)

    def load_gpt2_test(self, cls, gpt2_layer, input):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
        print("Input shape:", input.shape)
        output = layer(input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape)
        try:
            reference_output = gpt2_layer(input)
        except:
            reference_output = gpt2_layer(input, input, input)
            
        print("Reference output shape:", reference_output.shape, "\n")
        self.assertTrue(torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3), 
                        f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")


    def testLayerNorm(self):
        rand_float_test(LayerNorm, [2, 4, 768])
        load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])

    def testEmbed(self):
        rand_int_test(Embed, [2, 4])
        load_gpt2_test(Embed, reference_gpt2.embed, tokens)

    def testPosEmbed(self):
        rand_int_test(PosEmbed, [2, 4])
        load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)        

    def testAttention(self):
        rand_float_test(Attention, [2, 4, 768])
        load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

    def testMLP(self):
        rand_float_test(MLP, [2, 4, 768])
        load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])        

    def testTransformer(self):
        rand_float_test(TransformerBlock, [2, 4, 768])
        load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

    def testUnembed(self):
        rand_float_test(Unembed, [2, 4, 768])
        load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])        

    def testDemo(self):
        rand_int_test(DemoTransformer, [2, 4])
        load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

    def testGeneration(self):
        start_sequence = "Today I was walking home, when suddenly"
        start_tokens = reference_gpt2.to_tokens(start_sequence, prepend_bos=True)
        max_new_tokens = 20  # Maximum number of tokens to generate

        # Generate tokens using greedy decoding
        generated_tokens = greedy_decode(demo_gpt2, start_tokens, max_new_tokens)

        # Decode generated tokens back to text
        generated_text = reference_gpt2.to_string(generated_tokens[1:])
        print("Generated Text:", generated_text)

        reference_generation = reference_gpt2.generate(start_sequence,
                                                       max_new_tokens=max_new_tokens,
                                                       stop_at_eos=False,
                                                       do_sample=False)
        self.assertEqual(reference_generation, generated_text)
