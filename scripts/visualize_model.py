import spacy_streamlit
import typer


def main(models: str, default_text: str = "W sierpniu 2022 zatrucie pokarmowe. Potem utrzymywały się objawyy refluxu "
         "ż-p. Gdy stosowała Nolpazę objawy ustąpiły, a przy próbie odstawienia nawróciły odbijania.  Sporadycznie "
         "pojawiają się biegunki.  Wróciła do stosowania Nolpaza 40 mg. Nie pije kawy, nie pali, jest na diecie "
         "wegetariańskiej. Dolegliwości nie nasilają się po posiłkach, a wręcz się zmniejszają.  Wzrost: 165 cm Masa "
         "ciała: 57 kg (w dwa tygodnie schudła ok. 3-4 kg)"):
    models = [name.strip() for name in models.split(",")]
    spacy_streamlit.visualize(models, default_text, visualizers=["ner"])


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
