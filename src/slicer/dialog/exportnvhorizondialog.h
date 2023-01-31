#ifndef SRC_SLICER_DIALOG_EXPORTNVHORIZONDIALOG_H
#define SRC_SLICER_DIALOG_EXPORTNVHORIZONDIALOG_H

#include <QDialog>

class FreeHorizon;
class NvLineEdit;

class QListWidget;
class QListWidgetItem;

class ExportNVHorizonDialog : public QDialog {
	Q_OBJECT
public:
	ExportNVHorizonDialog(FreeHorizon* freeHorizon, QWidget* parent=0);
	~ExportNVHorizonDialog();

	QString getFilePath(const QString& attrName);
	QString getHorizonName();

	bool checkNameAndAttributsCompatibility();

public slots:
	virtual void accept() override;

private slots:
	void horizonDestroyed();
	void newNameChanged();
	void slotSelectItem(QListWidgetItem * current, QListWidgetItem * previous);

private:
	void run();

	QListWidget* createAttributsList();
	QListWidget* createTargetHorizonList();
	bool testSismageHorizon(const QString& sismageHorizonPath);

	FreeHorizon* m_freeHorizon;
	QString m_newSismageHorizonName;
	long m_selectedItem = -1;
	QString m_sismageHorizonPath;
	std::string m_surveyPath;

	QListWidget* m_attributsListWidget;
	NvLineEdit* m_newNameLineEdit;
	QListWidget* m_targetHorizonList;
};

#endif // SRC_SLICER_DIALOG_EXPORTNVHORIZONDIALOG_H
