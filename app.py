import gradio as gr
from pipeline import process_query   

demo= gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="❓ Ask your mental health question here", placeholder="e.g. How can I deal with anxiety about work?", lines=2),
    outputs=gr.Markdown(label="🩺 Top Therapist Answers"),
    title="🧘 CounselChat Q&A Assistant",
    description="Explain in brief about your problems.",
    allow_flagging="never"
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
