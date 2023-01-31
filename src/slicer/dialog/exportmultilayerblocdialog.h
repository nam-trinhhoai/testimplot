#ifndef EXPORTLAYERBLOCDIALOG_H
#define EXPORTLAYERBLOCDIALOG_H

#include <QDialog>

class QListWidgetItem;
class QListWidget;
class QLineEdit;
class QCheckBox;

class FixedRGBLayersFromDatasetAndCube;

class ExportMultiLayerBlocDialog : public QDialog {
Q_OBJECT
public:
	ExportMultiLayerBlocDialog( QString title, FixedRGBLayersFromDatasetAndCube* dataset, QWidget *parent = nullptr);
	~ExportMultiLayerBlocDialog();


	void accepted();

private:
	QLineEdit* m_firstGTLE;
	QLineEdit* m_stepGTLE;
	QLineEdit* m_lastGTLE;
	QLineEdit* m_nameLE;
	QCheckBox* m_interpolationCheckBox;

	FixedRGBLayersFromDatasetAndCube* m_refDataset;
};

#endif
