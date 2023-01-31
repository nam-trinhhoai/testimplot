#ifndef SRC_SLICER_DIALOG_MULTITEXTSELECTORDIALOG_H
#define SRC_SLICER_DIALOG_MULTITEXTSELECTORDIALOG_H

#include <QDialog>

class QListWidget;

class MultiTextSelectorDialog : public QDialog {
	Q_OBJECT
public:
	MultiTextSelectorDialog(QStringList texts, QWidget* parent=nullptr);
	virtual ~MultiTextSelectorDialog();

	QStringList selectedTexts();

private:
	QListWidget* m_listWidget;
};

#endif
