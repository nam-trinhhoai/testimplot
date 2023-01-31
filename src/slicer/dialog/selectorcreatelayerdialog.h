#ifndef SelectOrCreateLayerDialog_H_
#define SelectOrCreateLayerDialog_H_

#include <QDialog>
#include <QList>

class RgbLayerFromDataset;
class FixedLayerFromDataset;
class QComboBox;
class QLineEdit;
class QSpinBox;

// May need to be renamed or add abstraction for parameters
class SelectOrCreateLayerDialog : public QDialog {
public:
	SelectOrCreateLayerDialog(const QList<RgbLayerFromDataset*>& listRgb,
			const QList<FixedLayerFromDataset*>& listGray, QString defaultLabelPca,
			QString defaultLabelTmap);
	~SelectOrCreateLayerDialog();


	long layerIndex() const;
	bool isLayerNew() const;
	QString layer() const;
	QString label() const;

	long paramTmapSize() const;
	void setParamTmapSize(long);
	long paramExampleStep() const;
	void setParamExampleStep(long);
	long maxExampleStep() const;
	void setMaxExampleStep(long);

	bool isTmapChoosen() const;
	bool isPcaChoosen() const;


private slots:
	void newLayerLineEditChanged();
	void newLabelLineEditChanged();
	void changeLayerIndex(int index);
	void changeProcess(int index);

private:
	QList<RgbLayerFromDataset*> m_listRgb;
	QList<FixedLayerFromDataset*> m_listGray;
	QComboBox* m_layerComboBox;
	QComboBox* m_labelComboBox;
	QLineEdit* m_newLayerLineEdit;
	QLineEdit* m_newLabelLineEdit;
	QComboBox* m_processComboBox;

	QSpinBox* m_tmapSizeSpinBox;
	QSpinBox* m_exampleStepSpinBox;

	QString m_layer = "";
	QString m_label = "";
	QString m_defaultLabelTmap = "tmapLabel";
	QString m_defaultLabelPca = "pcaLabel";
	long m_index = -1;
	bool m_isLayerNew;

	long m_paramTmapSize = 30;
	long m_paramExampleStep = 10;
	long m_maxExampleStep = 1000;
	bool m_chooseTmap; // false imply pca
};

#endif
