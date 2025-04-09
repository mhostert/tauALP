#include "Pythia8/Pythia.h"
#include <iostream>
#include <cmath>
#include <random>

using namespace Pythia8;

int main(int argc, char* argv[]) {
  
  if (argc != 4) {
      std::cerr << "Usage: " << argv[0] << " <no of output events>  <output filename>  <energy of the proton beam>" << std::endl;
      return 1;
  }

  int nTauEvents = std::stoi(argv[1]);
  std::string baseFilename = argv[2];
  std::string Eproton_beam = argv[3];
  std::string outputFileName = "pythia8_events/tau_events_" + baseFilename + ".txt";

  // Create and open a file for output
  std::ofstream outFile(outputFileName);
  if (!outFile.is_open()) {
      std::cerr << "Error: Could not open file for writing." << std::endl;
      return 1;
  }

  // Add header to the output file
  outFile << "# event_number particle_count pid E px py pz mother_pid E_mother px_mother py_mother pz_mother" << std::endl;

  // Random number generator
  std::random_device rd; // Used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Mersenne Twister engine, a good choice for most cases
  std::uniform_real_distribution<double> unif(0.0, 1.0);


  // Generator
  Pythia pythia;

  // Setting up the Monash tune for p-p collisions
  pythia.readString("Tune:pp = 14");


  // Set up proton on proton at 120 GeV per beam (15.5 GeV collision energy)
  // pythia.readString("Beams:eCM = 15.5");  // Center of mass energy in GeV
  pythia.readString("Beams:eCM = "+Eproton_beam);  // Center of mass energy in GeV
  
  
  // Beam settings for one beam at rest and the other at 120 GeV
  pythia.readString("Beams:idA = 2212"); // proton
  pythia.readString("Beams:idB = 2212"); // proton
  pythia.readString("Beams:eA = "+Eproton_beam); // proton beam energy
  pythia.readString("Beams:eB = 0.0");   // proton at rest
  pythia.readString("Beams:frameType = 2"); // fixed target

  // Set the minimum pT for hard QCD (perturbative) processes (formally divergent...)
  pythia.readString("PhaseSpace:pTHatMin = 0.01");

  // Allow decays (including tau decays) to occur
  pythia.readString("ProcessLevel:all = on");
  pythia.readString("HadronLevel:Decay = on");
  
  // // Enable QCD processes which are necessary for hadron production
  pythia.readString("HardQCD:all = off");
  // pythia.readString("HardQCD:hardccbar = on");
  // pythia.readString("HardQCD:hardbbbar = on");

  // // Allow soft QCD, which is more appropriate at low energies
  pythia.readString("SoftQCD:all = on");
  pythia.readString("Onia:all = on");
  pythia.readString("SoftQCD:nonDiffractive = on");
  pythia.readString("SoftQCD:singleDiffractive = on");
  pythia.readString("SoftQCD:doubleDiffractive = on");

  //pythia.readString(“PhaseSpace:pTHatMin = 4.“);
  pythia.readString("431:onMode = off");
  pythia.readString("431:onIfAny = 15");
  pythia.readString("-431:onMode = off");
  pythia.readString("-431:onIfAny = -15");
  pythia.readString("411:onMode = off");
  pythia.readString("411:onIfAny = 15");
  pythia.readString("-411:onMode = off");
  pythia.readString("-411:onIfAny = -15");
  pythia.readString("100443:onMode = off");
  pythia.readString("100443:onIfAny = 15");
  pythia.readString("15:onMode = off");

  // Decrease verbosity
  pythia.readString("Init:showProcesses = off");   // Suppress process listing at initialization
  pythia.readString("Init:showMultipartonInteractions = off");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showAllSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberCount = 0");       // Disable periodic event listing
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  // Set error and warning verbosity
  pythia.readString("Print:quiet = on");           // Reduce printing of warnings
  pythia.settings.flag("Check:event", false);      // Disable extended event checking


  // Initialize the generator
  pythia.init();


  // Define counter for tau particles
  int tauCount = 0;
  int eventNumber = 0;

  while (tauCount < nTauEvents) {
    if (!pythia.next()) continue;
    
    for (int i = 0; i < pythia.event.size(); ++i) {
    
      int id = pythia.event[i].id();
    
      // Tau found
      if (abs(id) == 15) {

          ++tauCount;
          // Write particle data to the file with double precision
          outFile << std::fixed << std::setprecision(10)
            << eventNumber << " "
            << tauCount << " " 
            << id << " " 
            << pythia.event[i].e() << " " 
            << pythia.event[i].px() << " " 
            << pythia.event[i].py() << " " 
            << pythia.event[i].pz() << " "
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].id() : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].e() : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].px() : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].py() : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].pz() : 0) << std::endl;
          
        }

      }

    // Generated a new event
    ++eventNumber;

  }
  // Close the output file
  outFile.close();


  double totalCrossSection = pythia.info.sigmaGen();
  std::cout << "Total cross section of proton-proton collisions: " << totalCrossSection << " mb" << std::endl;


  return 0;
}
