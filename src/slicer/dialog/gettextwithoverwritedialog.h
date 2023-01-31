#ifndef SRC_SLICER_DIALOG_GETTEXTWITHOVERWRITEDIALOG_H
#define SRC_SLICER_DIALOG_GETTEXTWITHOVERWRITEDIALOG_H

#include <QDialog>

class QLineEdit;

class GetTextWithOverWriteDialog : public QDialog {
	Q_OBJECT
public:
	GetTextWithOverWriteDialog(const QString& prefix, QWidget* parent=0);
	virtual ~GetTextWithOverWriteDialog();

	QString text() const;
	void setText(const QString& text);

	bool isOverwritten() const;

public slots:
	void overwrite();

private:
	QLineEdit* m_textEdit;
	bool m_overwrite;
};

#endif
