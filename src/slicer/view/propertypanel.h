#ifndef PropertyPanel_H
#define PropertyPanel_H


#include <QDialog>
#include <QCheckBox>
#include <QLabel>
#include <QSlider>
#include <QLineEdit>
#include <QPalette>
#include <QPushButton>
#include <QGroupBox>
#include <QGridLayout>



class PropertyPanel: public QDialog
{
	Q_OBJECT
public:
	PropertyPanel(QWidget* parent);


	void init();

	void openIni();
	void saveIni();

	bool showInfos3d(){ return m_showInfos3d;}
	bool showGizmo3d(){ return m_showGizmo3d;}

	float speedHelico(){ return m_speedHelico;}
	float speedUpDown(){ return m_speedAltitude;}

signals:
	void simplifySurfaceChanged(int);

	void simplifySeuilWellChanged(double);

	void simplifySeuilLogsChanged(int);

	void wellDiameterChanged(int);
	void wellMapWidthChanged(double);
	void wellSectionWidthChanged(double);

	void pickDiameterChanged(int);
	void pickThicknessChanged(int);

	void logThicknessChanged(int);
	void colorLogChanged(QColor);

	void showInfo3DChanged(bool);
	void showGizmo3DChanged(bool);

	void speedUpDownChanged(float);
	void speedHelicoChanged(float);

	void speedRotHelicoChanged(float);

	void showNormalsWellChanged(bool);
	void wireframeWellChanged(bool);

	void colorWellChanged(QColor);
	void colorSelectedWellChanged(QColor);

	void speedMaxAnimChanged(int);
	void altitudeMaxAnimChanged(int);

	void showHelicoChanged(bool);

public slots:

void reset();

	void setSurfacePrecision(int value);
	void setSurfacePrecision();

	void setWellPrecision(int value);
	void setWellPrecision();

	void setLogsPrecision(int value);
	void setLogsPrecision();

	void setWellDiameter(int value);
	void setWellDiameter();

	void setWellMapWidth(int value);
	void setWellMapWidth();

	void setWellSectionWidth(int value);
	void setWellSectionWidth();

	void setPickDiameter(int value);
	void setPickDiameter();

	void setPickThickness(int value);
	void setPickThickness();

	void setLogThickness(int value);
	void setLogThickness();

	void setLogColor();

	void setSpeedUpDown(int value);
	void setSpeedUpDown();

	void setSpeedHelico(int value);
	void setSpeedHelico();

	void setSpeedRotHelico(int value);
	void setSpeedRotHelico();

	void setWellDefaultColor();
	void setWellSelectedColor();

	void info3dChecked(int);
	void gizmo3dChecked(int);

	void wireframeWellChecked(int );
	void showNormalsWellChecked(int );

	void setSpeedMaxAnim(int value);
	void setSpeedMaxAnim();

	void setAltitudeMaxAnim(int value);
	void setAltitudeMaxAnim();

	void showHelicoChecked(int);


private:

	QSlider* m_sliderPrecisionSurface;
	QLineEdit* m_editPrecisionSurface;

	QSlider* m_sliderPrecisionWell;
	QLineEdit* m_editPrecisionWell;

	QSlider* m_sliderMapWidthWell;
	QLineEdit* m_editMapWidthWell;

	QSlider* m_sliderSectionWidthWell;
	QLineEdit* m_editSectionWidthWell;

	QSlider* m_sliderDiameterWell;
	QLineEdit* m_editDiameterWell;

	QSlider* m_sliderPrecisionLogs;
	QLineEdit* m_editPrecisionLogs;

	QSlider* m_sliderSpeedHelico;
	QLineEdit* m_editSpeedHelico;


	QSlider* m_sliderSpeedRotHelico;
	QLineEdit* m_editSpeedRotHelico;

	QSlider* m_sliderUpDown;
	QLineEdit* m_editUpDown;

	QSlider* m_sliderDiameterPick;
	QLineEdit* m_editDiameterPick;

	QSlider* m_sliderThickness;
	QLineEdit* m_editThickness;

	QSlider* m_sliderThicknessLog;
	QLineEdit* m_editThicknessLog;

	QSlider* m_sliderSpeedMax;
	QLineEdit* m_editSpeedMax;

	QSlider* m_sliderAltitudeMax;
	QLineEdit* m_editAltitudeMax;

	QPushButton* m_buttonColorLog;

	QPushButton* m_buttonColorWell1;
	QPushButton* m_buttonColorWell2;

	QCheckBox* checkview1;
	QCheckBox* checkview2;
	QCheckBox* checkview3;
	QCheckBox* checkview4;
	QCheckBox* checkview5;

private:
	int m_wellPrecision = 2;
	int m_logsPrecision = 1;
	int m_logsThickness = 3;
	QColor m_logsColor = Qt::green;


	double m_wellMapWidth = 2.0;
	double m_wellSectionWidth = 2.0;
	double m_wellDiameter=30.0;
	QColor m_wellColor = Qt::yellow;
	QColor m_wellSelectedColor = Qt::cyan;

	float m_pickDiameter =50.0f;
	float m_pickThickness = 15.0f;

	float m_speedAltitude = 25.0f;
	float m_speedHelico = 10.0f;
	float m_speedRotHelico = 2.0f;

	int m_speedMaxAnim = 200;
	int m_altitudeMaxAnim = 400;

	int m_surfacePrecision=10;

	bool m_showInfos3d = true;
	bool m_showGizmo3d = true;

	bool m_wireframeWell = false;

	bool m_showNormalsWell = false;
	bool m_showHelico = false;
};


#endif
