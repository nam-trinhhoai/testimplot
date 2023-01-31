#include "nvlineedit.h"

#include <QRegularExpressionValidator>


NvLineEdit::NvLineEdit(const QString& contents, AllowedCharactersFlags flags, QWidget* parent) :
			QLineEdit(fix(contents, flags), parent), m_flags(flags) {
	QRegularExpressionValidator* val = createValidator(this);
	setValidator(val);
}

NvLineEdit::NvLineEdit(AllowedCharactersFlags flags, QWidget* parent) : QLineEdit(parent),
			m_flags(flags) {
	QRegularExpressionValidator* val = createValidator(m_flags, this);
	setValidator(val);
}

NvLineEdit::~NvLineEdit() {

}

QString NvLineEdit::allowedCharacters(AllowedCharactersFlags flags) {
	QString allowedChars = "a-zA-Z0-9";
	if (flags & AllowedCharactersFlag::Underscore) {
		allowedChars += "_";
	}
	if (flags & AllowedCharactersFlag::Dash) {
		allowedChars += "\\-";
	}
	if (flags & AllowedCharactersFlag::Plus) {
		allowedChars += "\\+";
	}
	if (flags & AllowedCharactersFlag::Dot) {
		allowedChars += "\\.";
	}
	if (flags & AllowedCharactersFlag::At) {
		allowedChars += "@";
	}
	if (flags & AllowedCharactersFlag::Parenthesises) {
		allowedChars += "\\(\\)";
	}
	if (flags & AllowedCharactersFlag::SquareBrackets) {
		allowedChars += "\\[\\]";
	}
	if (flags & AllowedCharactersFlag::CurlyBrackets) {
		allowedChars += "\\{\\}";
	}
	if (flags & AllowedCharactersFlag::Space) {
		allowedChars += " ";
	}

	return allowedChars;
}

QRegularExpressionValidator* NvLineEdit::createValidator(QObject* parent) {
	return createValidator(AllowedCharactersFlag::Default, parent);
}

QRegularExpressionValidator* NvLineEdit::createValidator(AllowedCharactersFlags flags, QObject* parent) {
	QString allowedChars = allowedCharacters(flags);

	QRegularExpression exp("^[" + allowedChars + "]+$");
	return new QRegularExpressionValidator(exp, parent);
}

QString NvLineEdit::fix(const QString& text, AllowedCharactersFlags flags) {
	QString allowedChars = allowedCharacters(flags);

	QRegularExpression exp("[^" + allowedChars + "]+");
	QRegularExpressionMatch match = exp.match(text);

	QString newText = text;
	while (match.isValid() && match.hasMatch()) {
		newText.replace(match.captured(0), "");
		int startPos = match.capturedStart(0);
		match = exp.match(newText, startPos);
	}
	return newText;
}
