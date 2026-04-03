import streamlit as st

st.title("AI Safety Agent Demo")

user_input = st.text_area("Enter text to evaluate")

def mock_evaluate(text):
    text = text.lower()

    if "hack" in text or "attack" in text:
        return {
            "judge_1": "unsafe",
            "judge_2": "unsafe",
            "judge_3": "unsafe",
            "final": "UNSAFE",
            "reason": "Potential harmful intent detected"
        }
    elif "insult" in text or "stupid" in text:
        return {
            "judge_1": "safe",
            "judge_2": "unsafe",
            "judge_3": "safe",
            "final": "BORDERLINE",
            "reason": "Disagreement among judges"
        }
    else:
        return {
            "judge_1": "safe",
            "judge_2": "safe",
            "judge_3": "safe",
            "final": "SAFE",
            "reason": "No safety concern"
        }

if st.button("Evaluate Safety"):
    result = mock_evaluate(user_input)

    st.subheader("Judge Results")
    st.write("Judge 1:", result["judge_1"])
    st.write("Judge 2:", result["judge_2"])
    st.write("Judge 3:", result["judge_3"])

    st.subheader("Final Decision")
    st.write(result["final"])

    st.subheader("Reason")
    st.write(result["reason"])