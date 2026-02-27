import { pipeline, TextStreamer, InterruptableStoppingCriteria, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.3';

// Optimizations for WebGPU
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;

let generator = null;
let stoppingCriteria = new InterruptableStoppingCriteria();

const MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX";
const DTYPE = "q4";

// Simple state tracking
let isGenerating = false;

self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    if (type === 'load') {
        if (generator) return;

        try {
            self.postMessage({ type: 'status', status: 'loading', message: 'Initializing model...' });

            generator = await pipeline('text-generation', MODEL_ID, {
                dtype: DTYPE,
                device: 'webgpu',
                progress_callback: (p) => {
                    self.postMessage({
                        type: 'progress',
                        file: p.file,
                        progress: p.progress,
                        status: p.status,
                        name: p.name
                    });
                }
            });

            self.postMessage({ type: 'status', status: 'ready', message: 'Model loaded successfully!' });
        } catch (err) {
            console.error(err);
            self.postMessage({ type: 'status', status: 'error', message: err.message || String(err) });
            generator = null;
        }
    }
    else if (type === 'generate') {
        if (!generator || isGenerating) return;

        isGenerating = true;
        stoppingCriteria.reset();

        const { messages, max_new_tokens = 2048, temperature = 0.7 } = data;

        try {
            self.postMessage({ type: 'status', status: 'generating' });

            let outputText = "";
            let tokenCount = 0;
            let startTime = performance.now();

            const streamer = new TextStreamer(generator.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: false,
                callback_function: (output) => {
                    if (output === "<|im_end|>") return;

                    outputText += output;
                    tokenCount++;

                    const elapsed = (performance.now() - startTime) / 1000;
                    const tps = tokenCount > 1 && elapsed > 0 ? (tokenCount - 1) / elapsed : 0;

                    self.postMessage({
                        type: 'chunk',
                        chunk: output,
                        tps: tps.toFixed(1)
                    });
                }
            });

            // Format messages for the pipeline
            const apiMessages = messages.map(m => ({ role: m.role, content: m.content }));

            await generator(apiMessages, {
                max_new_tokens: max_new_tokens,
                temperature: temperature,
                do_sample: temperature > 0,
                streamer,
                stopping_criteria: stoppingCriteria
            });

            self.postMessage({ type: 'complete' });
        } catch (err) {
            console.error("Generation error:", err);
            self.postMessage({ type: 'error', message: err.message || String(err) });
        } finally {
            isGenerating = false;
        }
    }
    else if (type === 'stop') {
        stoppingCriteria.interrupt();
    }
});
