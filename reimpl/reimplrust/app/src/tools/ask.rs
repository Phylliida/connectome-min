//! The ask_user tool: the model asks the user up to four questions, each with options to pick (one, or
//! several), and the user answers in the page, with an answer of their own and notes if they like, on
//! the card Claude Code's questions (AskUserQuestion, claude.rs) have too, whose input this takes. A
//! session has it while its settings' `questions` is on. Its calls don't run here: the session's thread
//! asks the pages (work.rs), however the session allows tools, since only the user can answer, and the
//! result is what the user answered (`told`), or that they declined.

use super::{Tool, Toolbox};
use connectome::{json::Value, obj};

pub const NAME: &str = "ask_user";

/// The result of questions the user declined to answer, here and in Claude Code's (claude.rs).
pub const DECLINED: &str = "The user declined to answer the questions.";

/// ask_user, declared.
pub fn tool() -> Toolbox {
    let text = |description: &str| obj! {"type" => "string", "description" => description};
    let option = obj! {"type" => "object", "required" => vec!["label", "description"], "properties" => obj! {
        "label" => text("What the user sees and picks: a few words"),
        "description" => text("What picking it means, or what follows from it"),
        "preview" => text("Something to see beside it, if comparing helps: a sketch, a snippet (monospace)"),
    }};
    let options = obj! {"type" => "array", "items" => option, "minItems" => 2, "maxItems" => 4, "description" => "2 to 4 distinct choices; the user can also write their own"};
    let question = obj! {"type" => "object", "required" => vec!["question", "header", "options"], "properties" => obj! {
        "question" => text("The question in full, ending with a question mark"),
        "header" => text("A short label shown beside it, 12 characters at most"),
        "options" => options,
        "multiSelect" => obj! {"type" => "boolean", "description" => "Whether the user may pick several (false by default)"},
    }};
    let description = "Ask the user one to four questions, each with options to pick from; they can also write their own answer and add notes. \
        Use it when you need their decision or something only they know. The call waits for their answers, which are its result.";
    let input_schema = obj! {"type" => "object", "required" => vec!["questions"], "properties" => obj! {"questions" => obj! {"type" => "array", "items" => question, "minItems" => 1, "maxItems" => 4}}};
    let definition = obj! {"name" => NAME, "description" => description, "input_schema" => input_schema};
    // asks: false, so no chip in the page's tools: its questions always go to the user (work.rs), never to the allow list
    Toolbox(vec![Tool { definition, asks: false, run: Box::new(|_, _| Err("ask_user's questions are the user's to answer".into())) }])
}

/// What the user `given` the questions ({answers, annotations}, by question), as the model hears it:
/// each question with its answer and the notes they added, as Claude Code words its own.
pub fn told(given: &Value) -> String {
    let answers = given["answers"].as_object().into_iter().flatten();
    let parts: Vec<String> = answers.map(|(question, answer)| {
        let notes = given["annotations"][question]["notes"].as_str().map_or(String::new(), |notes| format!(" notes: {notes}"));
        format!("\"{question}\"=\"{}\"{notes}", answer.as_str().unwrap_or(""))
    }).collect();
    if parts.is_empty() { "The user didn't answer the questions.".into() } else { format!("The user answered: {}.", parts.join(", ")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_are_told_as_claude_code_tells_them() {
        let given = obj! {"answers" => obj! {"Color?" => "Blue", "Pets?" => "Cats, Dogs"}, "annotations" => obj! {"Color?" => obj! {"notes" => "navy"}}};
        assert_eq!(told(&given), r#"The user answered: "Color?"="Blue" notes: navy, "Pets?"="Cats, Dogs"."#);
        assert_eq!(told(&obj! {}), "The user didn't answer the questions.");
    }
}
