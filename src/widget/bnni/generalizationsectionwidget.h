#ifndef GENERALIZATIONSECTIONWIDGET_H
#define GENERALIZATIONSECTIONWIDGET_H

#include <kddockwidgets/MainWindow.h>

#include <QWidget>
#include <QString>
#include <QVector>
#include <QMap>
#include <QTimer>
#include <QDoubleSpinBox>
#include <QFileSystemWatcher>

#include <chrono>
#include <utility>

#include "Canvas2dFullSync.h"
#include "MuratCanvas2dFullView.h"
#include "Matrix2D.h"
#include "palettewidget.h"
#include "paletteholder.h"
#include "structures.h"
#include "wellpolylinemanager.h"

class QComboBox;
class QDoubleSpinBox;
class QFormLayout;
class QGroupBox;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSlider;
class QSpinBox;
class QTreeWidget;

namespace KDDockWidgets {
class DockWidget;
}


class GeneralizationSectionWidget : public KDDockWidgets::MainWindow
{
    Q_OBJECT

public:
    explicit GeneralizationSectionWidget(QWidget *parent = 0);
    ~GeneralizationSectionWidget();

    void setSurvey(QString txt); // to remove
    void setYStep(int);
    void setSuffix(QString);
    void setScales(QVector<float>);
    void setPredictSampleRate(float predictSampleRate);
    void setWells(QVector<Well>* wells);
    void launchGeneralization();
    void setSeismicDynamic(QVector<std::pair<float, float>> array);
    void setLogDynamic(QVector<std::pair<float, float>> array);
    void setWorkDir(QString workDir);
    void setIJKDirectory(QString ijk_dir);
    void setSelectedLogsIndexes(QVector<int>);
    void setNetwork(NeuralNetwork network);
    void setTrainSampleRate(float trainSampleRate);

private:
    static long windowNextId;

    typedef struct Viewer {
        MuratCanvas2dFullView* view = nullptr;
        KDDockWidgets::DockWidget* dockWidget = nullptr;
        Matrix2DLine<float>* mat = nullptr;
        bool isSeismic = false; // true for seismic, false for well kind
        QString file; // seismic3d file for seismic, checkpoint prefix for well kind
        int iter_number = -1;
        PaletteWidget* paletteWidget = nullptr;
    } Viewer;

    void orderedInsertion(QVector<Viewer>& array, Viewer& item);
    void setScaleUI();
    int sliderValueToVolumeIndex();
    void loadJsonWell(const QString& wellName);
    void unloadJsonWell(const QString& wellName);
    void setupGui();
    QVector<Viewer>::const_iterator getFirstGUIValidCheckPointViewer() const;
    QString formatCheckpointName(const QString& checkPointFile);
    //void findIJKDirectory();

    QString saveSuffix = "tmp";
    int yStep = 1;
    QString workDir;

    QStringList loadedJsonWells; // to remember which wells were loaded from json
    QVector<QDoubleSpinBox*> scalesEdit;
    QVector<float> scales;
    QVector<std::pair<QString, QString>> seismicNames;
    QVector<std::pair<PaletteWidget*, PaletteHolder*>> seismicPalette;
    QVector<QString> logNames;
    QVector<std::pair<PaletteWidget*, PaletteHolder*>> logPalette;

    QVector<std::pair<float, float>> logDynamic;
    QVector<std::pair<float, float>> seismicDynamic;

    QVector<Well>* wells = nullptr;
    QVector<QString> trajectories;

    float seismicSampleRate = 1;
    float trainSampleRate = 1;
    float predictSampleRate = 1;
    float currentPredictSampleRate = 1; // because sample rate can change during during a computation
    bool m_currentPredictSampleRateSet = false;

    bool hasBeenReset = true;

    QVector<Viewer> viewers;
    Canvas2dFullSync synchronizer;

    int dimx = 0;
    int dimy = 0;
    int dimz = 0;
    int stepy = 1;
    int stepz = 1;
    int originy = 0;
    int originz = 0;

    QVector<QPoint> randomPts;
    SectionOrientation orientation = SectionOrientation::INLINE;

    QString survey;

    // bench
    long initialDisplay = 0;
    long compute = 0;
    long finalDisplay = 0;
    std::chrono::time_point<std::chrono::steady_clock> compute_t1;
    std::chrono::time_point<std::chrono::steady_clock> compute_t2;

    QFileSystemWatcher fsWatcher;

    std::list<QString> stack;
    WellPolylineManager wellGraphicManager;

    QString IJK_dir = "";

    NeuralNetwork network = NeuralNetwork::NoNetwork;

    QVector<int> selectedLogsIndexes;
    QString m_uniqueName;

    // gui
    KDDockWidgets::DockWidget* m_checkPointsDockWidget;
    KDDockWidgets::DockWidget* m_ijkWellDockWidget;
    KDDockWidgets::DockWidget* m_jsonWellDockWidget;
    KDDockWidgets::DockWidget* m_paramsDockWidget;
    KDDockWidgets::DockWidget* m_runDockWidget;

    QListWidget* m_checkpointListWidget;
    QListWidget* m_ijkWellListWidget;
    QTreeWidget* m_jsonWellTreeWidget;

    QComboBox* m_orientationComboBox;
    QComboBox* m_randomComboBox;
    QSlider* m_sliceSlider;
    QSpinBox* m_sliceSpinBox;
    QDoubleSpinBox* m_predictSampleRateSpinBox;
    QLineEdit* m_suffixLineEdit;
    QSpinBox* m_yStepSpinBox;
    QGroupBox* m_scaleWidget;
    QFormLayout* m_scaleWidgetLayout;
    QPushButton* m_updateButton;

signals:
    void ystepChanged(int ystep);
    void saveSuffixChanged(QString);
    void scalesChanged(const QVector<float>&);
    void predictSampleRateChanged(float predictSampleRate);
    void generalize(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, QString checkpoint, QString suffix);
    void generalizeRandom(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, QString checkpoint, QString suffix, QStringList data, QString generalizationDir);

public slots:
    void setWellKind(QVector<QString> array);
    void setSeismicNames(QVector<std::pair<QString, QString>> array);
    void generalizationFinished(QString);
    void generalizationFailed(QString);
    void generalizationRefused(QString);
    void reset();
    void updateViewerPalette(PaletteWidget* palette);

private slots:
    void updateSaveSuffix(QString s);
    void updateYStep(int i);
    void updateScales(QVector<float>);
    void updateDimensions();
    void changeOrientation(QString);
    void updateCheckpointListWidget();
    void updateViewerListFromSelection();
    void predictFromStack();
    void setPredictSampleRateInternal(double predictSampleRate);
    void jsonWellSelectionChanged();
    void jsonWellModelDataChanged(const QModelIndex& topLeft,
            const QModelIndex& bottomRight, const QVector<int>& roles);

};

#endif // GENERALIZATIONSECTIONWIDGET_H
