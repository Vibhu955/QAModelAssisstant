import gradio as gr
from pipeline import process_query   # or from pipeline import process_query if you renamed

gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="❓ Ask your mental health question here", placeholder="e.g. How can I deal with anxiety about work?", lines=2),
    outputs=gr.Markdown(label="🩺 Top Therapist Answers"),
    title="🧘 CounselChat Q&A Assistant",
    description="Explain in brief about your problems.",
    allow_flagging="never"
).launch()
