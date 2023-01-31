#ifndef ComposeSeismicToRgbDialog_H_
#define ComposeSeismicToRgbDialog_H_

#include <QDialog>
#include <QList>
#include <QString>

class WorkingSetManager;
class SeismicSurvey;
class Seismic3DAbstractDataset;

class QComboBox;
class QSpinBox;
class QGridLayout;

class ComposeSeismicToRgbDialog : public QDialog {
	Q_OBJECT
public:
	ComposeSeismicToRgbDialog(WorkingSetManager* workingSet);
	ComposeSeismicToRgbDialog(SeismicSurvey* survey);
	~ComposeSeismicToRgbDialog();

	Seismic3DAbstractDataset* red() const;
	int channelRed() const;
	Seismic3DAbstractDataset* green() const;
	int channelGreen() const;
	Seismic3DAbstractDataset* blue() const;
	int channelBlue() const;
	Seismic3DAbstractDataset* alpha() const;
	int channelAlpha() const;

private:
	void initObject();
	void fillComboBox(QComboBox* comboBox);
	void setupChannel(const QString& name, QComboBox* comboBox, QSpinBox* spinBox, QGridLayout* gridLayout, int row);
	void changeRed(int index);
	void changeGreen(int index);
	void changeBlue(int index);
	void changeAlpha(int index);

	void changeRedChannel(int channel);
	void changeGreenChannel(int channel);
	void changeBlueChannel(int channel);
	void changeAlphaChannel(int channel);

	QComboBox* m_comboBoxRed, *m_comboBoxGreen, *m_comboBoxBlue, *m_comboBoxAlpha;
	QSpinBox* m_spinBoxRed, *m_spinBoxGreen, *m_spinBoxBlue, *m_spinBoxAlpha;


	Seismic3DAbstractDataset* m_red, *m_green, *m_blue, *m_alpha;
	int m_channelRed, m_channelGreen, m_channelBlue, m_channelAlpha;

	QList<Seismic3DAbstractDataset*> m_allDatasets;
};

#endif
