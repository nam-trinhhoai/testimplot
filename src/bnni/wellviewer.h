#ifndef WELLVIEWER_H
#define WELLVIEWER_H

#include <QWidget>
#include <QListWidget>
#include <QListWidgetItem>

#include "structures.h"
#include "Matrix2D.h"
#include "Canvas2dVerticalSync.h"
#include <memory>

class QVBoxLayout;
class QGridLayout;
class MuratCanvas2dLogView;
class LogStats;

namespace Ui {
class WellViewer;
}

class WellViewer : public QWidget
{
    Q_OBJECT

public:
    explicit WellViewer(QWidget *parent = 0);
    ~WellViewer();

    void updateWells(QVector<Well >* wells);
    void selectLogs(QVector<unsigned int> logsIndexes, QVector<LogParameter> logsParameters);
    void selectSeismics(QVector<unsigned int> seismicsIndexes, QVector<Parameter> seismicsParameters);
    bool setLogPreprocessing(QVector<int>);

    // search plateaus in well ranges for LogSample indexes in logIndexSearch
    // logIndexSearch default is search for all indexes
    static void changeRangeToRemoveWellPlateaus(Well& well, const std::vector<int>& logIndexSearch = std::vector<int>());
    static std::pair<bool, int> searchBeginPlateau(LogSample* tab, int size, const std::vector<int>& logIndexSearch, int plateauMinimumSize);
    static std::pair<bool, int> searchBeginPlateau(const float* tab, int size, int plateauMinimumSize);
    static std::pair<bool, int> searchEndPlateau(LogSample* tab, int size, const std::vector<int>& logIndexSearch);

    static bool isSeismicWithPlateau(const QVector<float>& seismicValues, int nVolume, int seismicWindowSize) ;

    static void removeWellsWithConstantSeismic(Well& well, int nVolume, int seismicWindowSize);

signals:
    void toggleWell(Well& well, bool show);
    void minMaxIndexChanged(int);

public slots:
    void wellChange(QListWidgetItem* item);
    void wellSelectedChanged();
    void addNewRange();
    void removeRange();


private:
    void reset();
    // posLines are the log view indexes
    void linesChanged(int viewIdx, const std::vector<double>& posLines);

    QVBoxLayout* holder = nullptr;
    Ui::WellViewer *ui;
    QVector<Well >* wells = nullptr;
    QVector<unsigned int> logsIndexes;
    QVector<MuratCanvas2dLogView*> logsViews;
    QVector<LogStats*> logsStats;
    QGridLayout* viewLayout = nullptr;
    QVector<std::shared_ptr<Matrix2DLine<double>>> mats;
    QVector<LogParameter> logsParameters;
    Canvas2dVerticalSync synchroniser;
    QVector<int> logPreprocessing;

    int currentWellIndex = -1;
};

#endif // WELLVIEWER_H
