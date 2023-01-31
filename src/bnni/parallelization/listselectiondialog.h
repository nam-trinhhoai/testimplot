#ifndef SRC_BNNI_PARALLELIZATION_LISTSELECTIONDIALOG_H
#define SRC_BNNI_PARALLELIZATION_LISTSELECTIONDIALOG_H

#include <QDialog>
#include <QStringList>

class QListWidget;

class ListSelectionDialog : public QDialog {
	Q_OBJECT
public:
	typedef struct SelectionItem {
		QString str;
		bool isSelected;
	} SelectionItem ;

	ListSelectionDialog(const QStringList& list, const QString& question);
	~ListSelectionDialog();

	// contruct in the same order as constructor lists
	const std::vector<SelectionItem> getList() const;

public slots:
	void selectAll();
	void deselectAll();

private slots:
	void selectionChanged();

private:
	std::vector<SelectionItem> m_list;
	QString m_question;

	QListWidget* m_listWidget;
};

#endif
