#ifndef STATUS_BAR_H
#define STATUS_BAR_H

#include <QWidget>
#include <QString>
class QLineEdit;
class QLabel;

class StatusBar: public QWidget {
public:
	StatusBar(QWidget *parent);
	virtual ~StatusBar();

	void setWorldCoordinateLabels(const QString &labelX, const QString &labelY);

	void x(double x);
	void y(double x);

	void i(int x);
	void j(int x);
	void depth(double x);

	void value(const QString &value);

	void clearI();
	void clearJ();
	void clearValue();
	void clearDepth();

private:
	QLineEdit *m_x;
	QLineEdit *m_y;
	QLineEdit *m_i;
	QLineEdit *m_j;
	QLineEdit *m_depth;
	QLineEdit *m_value;

	QLabel * m_labelWorldX;
	QLabel * m_labelWorldY;
};

#endif
