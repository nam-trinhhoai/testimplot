#ifndef WELLPOLYLINEMANAGER_H
#define WELLPOLYLINEMANAGER_H

#include <QObject>
#include <QVector>
#include <QVector3D>
#include <QDialog>

#include <memory>

#include "MuratCanvas2dFullView.h"
#include "structures.h"

class QGraphicsPathItem;

typedef struct PolylineViewer{
    QGraphicsPathItem* only_supports = nullptr;
    QGraphicsPathItem* supports = nullptr;
    QGraphicsPathItem* values = nullptr;
    MuratCanvas2dFullView* view = nullptr;
    std::shared_ptr<QMetaObject::Connection> connection;

} PolylineViewer;

class WellPolylineManager : public QObject
{
    Q_OBJECT
public:
    explicit WellPolylineManager(QObject *parent = 0);
    virtual ~WellPolylineManager();
    void updateViewers();
    long getWellProjectionMaxDistance();
    void setWellProjectionMaxDistance(long);
    void setLogDynamic(QVector<std::pair<float,float>> logDynamic);

    double getCumulFactor();

    void runGraphicsSettingsDialog(QWidget* dialogParent);
    void setValueIndex(int valueIndex);
    int getValueIndex();

    bool containsWell(const QString& name);

    // newRatio is in ]0; +inf[
    // if newRatio < 1, wells will be shrinked
    // if newRatio > 1, wells will grow
    void setRatioBetweenWellsAndImage(double newRatio);

signals:

public slots:
    void connectViewer(MuratCanvas2dFullView* viewer);
    void disconnectViewer(MuratCanvas2dFullView* viewer);
    void setOrientation(SectionOrientation orientation);
    void setRandomLine(const QVector<QPoint>& random_line);
    void addWell(const Well& well, QVector<QVector3D> wholeTrajectory=QVector<QVector3D>());
    void removeWell(QString name);
    void setSlice(long slice);

private slots:
    void changeScale(double factorX, double factorY);

private:
    void clearDisplay();
    void computeSupportsAndValues();
    void initPens(PolylineViewer polyViewer);

    QVector<Well> wells;
    QVector<QVector<QVector3D>> wholeTrajectories;
    QVector<PolylineViewer> viewers;

    SectionOrientation orientation = SectionOrientation::INLINE;
    QVector<QPoint> randomPts;
    QVector<QVector<QPolygonF>> supports;
    QVector<QVector<QPolygonF>> values;
    long slice=0;
    long well_projection_max_distance = 5;
    QVector<std::pair<float, float>> logDynamic;

    double pen_width = 3.0;
    QColor color_full_well_support = QColor(255, 0, 0);
    QColor color_full_well_value = QColor(255, 255, 255);
    double pen_width_support_only = 2.0;
    QColor color_support_only = QColor(255, 255, 0);

    double cumul_factorX = 1.0;
    double cumul_factorY = 1.0;
    double size_min_max = 500.0;
    double offset_min = 0.0;

    double ratioBetweenWellsAndImage = 1.0;

    int valueIndex = 0;
};

class WellPolylineManagerSettingsDialog : public QDialog
{
public:
	explicit WellPolylineManagerSettingsDialog(double pen_width, double pen_width_support_only, double size_min_max, double offset_min, QWidget* parent=0);
	~WellPolylineManagerSettingsDialog();

	double getPenWidth();
	double getPenWidthSupportOnly();
	double getSizeMinMax();
	double getOffsetMin();
private:
	double pen_width;
	double pen_width_support_only;
	double size_min_max;
	double offset_min;
};

#endif // WELLPOLYLINEMANAGER_H
