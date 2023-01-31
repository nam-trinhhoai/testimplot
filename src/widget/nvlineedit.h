#ifndef SRC_WIDGET_NVLINEEDIT_H
#define SRC_WIDGET_NVLINEEDIT_H

#include <QFlags>
#include <QLineEdit>

class QRegularExpressionValidator;

/**
 * This class is intended to be used to request text that may end up as file or directory name
 *
 * The validator will always allow letters from a to z, A to Z
 */
class NvLineEdit : public QLineEdit {
	Q_OBJECT
public:
	// the enum need to be in sync with function allowedCharacters()
	enum AllowedCharactersFlag {
		None = 0x0,
		Underscore = 0x1,
		Dash = 0x2,
		Plus = 0x4,
		Dot = 0x8,
		At = 0x10,
		Parenthesises = 0x20,
		SquareBrackets = 0x40,
		CurlyBrackets = 0x80,
		Space = 0x100,
		Default = Underscore | Dash,
		All = 0xffffffff
	};
	Q_DECLARE_FLAGS(AllowedCharactersFlags, AllowedCharactersFlag)

	NvLineEdit(const QString& contents, AllowedCharactersFlags flags=AllowedCharactersFlag::Default, QWidget* parent=nullptr);
	NvLineEdit(AllowedCharactersFlags flags=AllowedCharactersFlag::Default, QWidget* parent=nullptr);
	~NvLineEdit();

	static QString allowedCharacters(AllowedCharactersFlags flags);
	static QRegularExpressionValidator* createValidator(QObject* parent=nullptr);
	static QRegularExpressionValidator* createValidator(AllowedCharactersFlags flags, QObject* parent=nullptr);

	// remove not allowed characters
	static QString fix(const QString& text, AllowedCharactersFlags flags=AllowedCharactersFlag::Default);

private:
	AllowedCharactersFlags m_flags;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(NvLineEdit::AllowedCharactersFlags)

#endif // SRC_WIDGET_NVLINEEDIT_H
