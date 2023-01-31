#ifndef WellBorePropPanelOnSlice_H
#define WellBorePropPanelOnSlice_H

#include <QWidget>

class WellBore;
class QComboBox;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;

class WellBorePropPanelOnSlice : public QWidget {
	Q_OBJECT
public:
	WellBorePropPanelOnSlice(WellBore* data, QWidget* parent=0);
	~WellBorePropPanelOnSlice();

	long origin() const;
	long width() const;
	double logMin() const;
	double logMax() const;
	bool base() const;
	double logBase() const;
	bool fillTop() const;
	QColor fillTopColor() const;
	bool fillBottom() const;
	QColor fillBottomColor() const;

signals:
	void logChanged(long index);
	void colorChanged(QColor color);
	void originChanged(int origin);
	void widthChanged(int width);
	void logMinChanged(double mini);
	void logMaxChanged(double maxi);
	void baseChanged(bool activated);
	void logBaseChanged(double base);
	void fillTopChanged(bool activated);
	void fillTopColorChanged(QColor color);
	void fillBottomChanged(bool activated);
	void fillBottomColorChanged(QColor color);

public slots:
	void logLabelClick();
	void updateLog();
	void updateWidget();
	void setLog(long index);
	void setColor(QColor color);
	void setOrigin(int origin);
	void setWidth(int width);
	void setLogMin(double mini);
	void setLogMax(double maxi);
	void setBase(bool activated);
	void setLogBase(double base);
	void setFillTop(bool activated);
	void setFillTopColor(QColor color);
	void setFillBottom(bool activated);
	void setFillBottomColor(QColor color);

private slots:
	void logChangedInternal(int index);
	void colorChangedInternal();
	void originChangedInternal(int origin);
	void widthChangedInternal(int width);
	void logMinChangedInternal(double mini);
	void logMaxChangedInternal(double maxi);
	void baseChangedInternal(int state);
	void logBaseChangedInternal(double base);
	void fillTopChangedInternal(int state);
	void fillTopColorChangedInternal();
	void fillBottomChangedInternal(int state);
	void fillBottomColorChangedInternal();

private:
	void buildWidget(); // only call once in constructor

	WellBore* m_data;
	QLabel *m_logLabel;
	QPushButton *m_bpLog;
	QComboBox* m_logComboBox;
	QPushButton* m_colorHolder;
	QSpinBox* m_originSpinBox;
	QSpinBox* m_widthSpinBox;
	QDoubleSpinBox* m_logMinSpinBox;
	QDoubleSpinBox* m_logMaxSpinBox;
	QCheckBox* m_baseCheckBox;
	QDoubleSpinBox* m_logBaseSpinBox;
	QLabel* m_logBaseLabel;
	QCheckBox* m_fillTopCheckBox;
	QPushButton* m_fillTopColorHolder;
	QLabel* m_fillTopColorLabel;
	QCheckBox* m_fillBottomCheckBox;
	QPushButton* m_fillBottomColorHolder;
	QLabel* m_fillBottomColorLabel;

	QString m_currentLogName = "";
	QString m_currentLogPath = "";

	long m_noLogIndex;
	long m_origin;
	long m_width;
	double m_logMin;
	double m_logMax;
	bool m_base;
	double m_logBase;
	bool m_fillTop;
	QColor m_fillTopColor;
	bool m_fillBottom;
	QColor m_fillBottomColor;
	int getLogNameIndex(QString name);

};

#endif
