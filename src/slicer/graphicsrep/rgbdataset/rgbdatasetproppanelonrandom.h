#ifndef RGBDATASETPROPPANELONRANDOM_H_
#define RGBDATASETPROPPANELONRANDOM_H_

#include <QWidget>
#include <map>

class QComboBox;
class QSlider;
class QSpinBox;
class QGridLayout;
class QGroupBox;
class QLabel;
class QLineEdit;

class RgbDatasetRepOnRandom;
class Seismic3DAbstractDataset;
class SeismicSurvey;
class PaletteWidget;

class RgbDatasetPropPanelOnRandom : public QWidget {
	Q_OBJECT
public:
	RgbDatasetPropPanelOnRandom(RgbDatasetRepOnRandom* rep, QWidget* parent=0);
	~RgbDatasetPropPanelOnRandom();
	void updatePalette();

private slots:
	void redChanged();
	void greenChanged();
	void blueChanged();
	void alphaChanged();
	void constantAlphaChanged();
	void radiusAlphaChanged();

	void redChangedInternal(int index);
	void greenChangedInternal(int index);
	void blueChangedInternal(int index);
	void alphaChangedInternal(int index);

	void redChannelChangedInternal(int index);
	void greenChannelChangedInternal(int index);
	void blueChannelChangedInternal(int index);
	void alphaChannelChangedInternal(int index);

	void constantAlphaChangedInternal(int value);
	void radiusAlphaChangedInternal(int value);

	void dataAdded(Seismic3DAbstractDataset* data);
	void dataRemoved(Seismic3DAbstractDataset* data);

	void updateSynchroneSliders();
	void setRedIndex(int value);
	void setBlueIndex(int value);
	void setGreenIndex(int value);
	void frequencyChanged();

private:
	void initSynchroneSliders();
	void setupChannel(const QString& name, QComboBox* comboBox, QSpinBox* spinBox, QGridLayout* gridLayout, int row);
	void fillComboBox(QComboBox* comboBox);
	void initAllDatasets();
	bool removeDataFromComboBox(QComboBox* comboBox, QSpinBox* spinBox, int key);
	void createlinkedSliderSpin(QWidget *parent, QSlider *slider, QSpinBox *spin, QLineEdit *lineEdit);
	void updateRGBSynchroneSliders();
	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin, QLineEdit* lineEdit);
	void updateSliderSpin(int min, int max, QSlider *slider, QSpinBox *spin);


	std::size_t nextIndex() const;

	QComboBox* m_redComboBox;
	QSpinBox* m_redChannelSpinBox;
	QComboBox* m_greenComboBox;
	QSpinBox* m_greenChannelSpinBox;
	QComboBox* m_blueComboBox;
	QSpinBox* m_blueChannelSpinBox;
	QComboBox* m_alphaComboBox;
	QSpinBox* m_alphaChannelSpinBox;
	QSlider* m_constantAlphaSlider; // if alpha is None
	QSlider* m_radiusAlphaSlider; // if alpha is function of rgb ratio.
	QLabel* m_constantAlphaLabel;
	QLabel* m_radiusAlphaLabel;

	PaletteWidget *m_redPalette;
	PaletteWidget *m_greenPalette;
	PaletteWidget *m_bluePalette;

	QGroupBox* m_transparencyGroupBox;

    // synchrone Sliders objects
	QGroupBox* m_synchroneSliderGroupBox;

	QSlider *m_redSlider;
	QSpinBox *m_redSpin;
	QLineEdit *m_redLineEdit;

	QSlider *m_greenSlider;
	QSpinBox *m_greenSpin;
	QLineEdit *m_greenLineEdit;

	QSlider *m_blueSlider;
	QSpinBox *m_blueSpin;
	QLineEdit *m_blueLineEdit;

	long m_oldDeltaRed = 0;
	long m_oldDeltaBlue = 0;

	RgbDatasetRepOnRandom* m_rep;
	std::map<std::size_t, Seismic3DAbstractDataset*> m_allDatasets;
	SeismicSurvey* m_seismicSurvey;
	mutable std::size_t m_nextIndex = 0;
};

#endif
