
#ifndef RGBPALETTEWIDGET_H_
#define RGBPALETTEWIDGET_H_

#include <QWidget>

#include <QVector>
#include <vector>

class QLineEdit;
class QSlider;

class RangeAndHistogramWidget;
class IPaletteHolder;

class RGBPaletteWidget : public QWidget{
	Q_OBJECT
	Q_PROPERTY(float opacity READ getOpacity WRITE setOpacity NOTIFY opacityChanged)
public:
	RGBPaletteWidget(QWidget* parent = 0, Qt::WindowFlags f = Qt::Widget );
	virtual ~RGBPaletteWidget();

	void setPaletteHolders(QVector<IPaletteHolder *>image);
	void setPaletteHolder(int i,IPaletteHolder* image);

	QVector2D getRange(int i);

    float getOpacity() const;
public slots:
	void setOpacity(float val);
	void setRange(unsigned int i,const QVector2D &range);
signals:
	void opacityChanged(float opacity);
	void rangeChanged(unsigned int i,const QVector2D & range);

private slots:
	void opacityChanged(int value);

	void redRangeChanged(const QVector2D & range);
	void greenRangeChanged(const QVector2D & range);
	void blueRangeChanged(const QVector2D & range);
private:
	QVector<RangeAndHistogramWidget *> m_histoWidget;
	QSlider *m_opacity;
};

#endif
