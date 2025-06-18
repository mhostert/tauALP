#include "Pythia8/Pythia.h"
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cmath>
#include <random>

using namespace Pythia8;


int main(int argc, char* argv[]) {
  
  if (argc != 7) {
      std::cerr << "Usage: " << argv[0] << " <no of output events>  <output filename>  <energy of the proton beam> <soft-hard flag> <pTcut> <seed>" << std::endl;
      return 1;
  }

  #ifdef PYTHIA8_VERSION
    std::cout << "PYTHIA8_VERSION is defined as: " << PYTHIA_VERSION_INTEGER << std::endl;
  #else
    std::cout << "PYTHIA8_VERSION is NOT defined!" << std::endl;
  #endif

  int nTauEvents = std::stoi(argv[1]);
  std::string baseFilename = argv[2];
  std::string Eproton_beam = argv[3];
  std::string soft_or_hard = argv[4];
  std::string pTcut = argv[5];
  double pTcut_d = std::stod(pTcut);
  std::string seed = argv[6];
  std::string outputFileName = "pythia8_events/" + soft_or_hard + "_" + baseFilename + ".txt";

  // Create and open a file for output
  std::ofstream outFile(outputFileName);
  if (!outFile.is_open()) {
      std::cerr << "Error: Could not open file for writing." << std::endl;
      return 1;
  }

  // Generator
  Pythia pythia;

  // Get current time in nanoseconds since epoch
  auto now = std::chrono::high_resolution_clock::now();
  auto seed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

  // Create random number generator
  // std::mt19937_64 rng(seed_time);
  // std::uniform_int_distribution<int> dist(1, 900000000);

  // int seed = dist(rng);

  pythia.readString("Random:seed = " + seed);
  pythia.readString("Random:setSeed = on");
  
  // Setting up the Monash tune for p-p collisions
  pythia.readString("Tune:pp = 14");

  // Set up proton on proton at 120 GeV per beam (15.5 GeV collision energy)
  pythia.readString("Beams:eCM = " + Eproton_beam);  // Center of mass energy in GeV
  
  // Beam settings for one beam at rest and the other at 120 GeV
  pythia.readString("Beams:idA = 2212"); // proton
  pythia.readString("Beams:idB = 2212"); // proton
  // pythia.readString("Beams:eA = " + Eproton_beam); // proton beam energy
  // pythia.readString("Beams:eB = 0.0");   // proton at rest
  // pythia.readString("Beams:frameType = 2"); // fixed target

  if (soft_or_hard == "soft") {

    pythia.readString("HardQCD:all = off");
    pythia.readString("SoftQCD:all = on");
    pythia.readString("Onia:all = on");
    
    pythia.readString("SoftQCD:nonDiffractive = on");
    pythia.readString("SoftQCD:singleDiffractive = on");
    pythia.readString("SoftQCD:doubleDiffractive = on");
    pythia.readString("PhaseSpace:pTHatMin = 0.01"); // Minimum pT for hard processes in GeV

  } else if (soft_or_hard == "hard") {
      
      pythia.readString("HardQCD:all = on");
      pythia.readString("SoftQCD:all = off");
      pythia.readString("Onia:all = on");
      
      pythia.readString("HardQCD:hardccbar = on");
      pythia.readString("HardQCD:hardbbbar = on");
      pythia.readString("PhaseSpace:pTHatMin =" + pTcut); // Minimum pT for hard processes in GeV
      
      
  } else if (soft_or_hard == "both") { // Likely to lead to double counting, but useful for cross-checks
        
      pythia.readString("HardQCD:all = on");
      pythia.readString("SoftQCD:all = on");
      pythia.readString("Onia:all = on");
      pythia.readString("PhaseSpace:pTHatMin = 0.01");
      
      
  }
  else {
    std::cerr << "Error: Invalid soft/hard flag. Use 'soft', 'hard', or 'both'." << std::endl;
    return 1;
  }

  // Allow decays (including tau decays) to occur
  pythia.readString("ProcessLevel:all = on");
  pythia.readString("HadronLevel:Decay = on");

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

  // Write simulation configuration to header
  outFile << "# Simulation parameters:" << std::endl;
  outFile << "#   Pythia8 version: " << PYTHIA_VERSION_INTEGER << std::endl;
  outFile << "#   Beam energy (GeV): " << Eproton_beam << std::endl;
  outFile << "#   pT cut (GeV): " << pTcut << std::endl;
  outFile << "#   Random seed: " << seed << std::endl;
  outFile << "#   SoftQCD enabled: " << (pythia.flag("SoftQCD:all") ? "yes" : "no") << std::endl;
  outFile << "#   HardQCD enabled: " << (pythia.flag("HardQCD:all") ? "yes" : "no") << std::endl;
  outFile << "# --------------------- " << std::endl;  
  outFile << "# event_number particle_count pid E px py pz mother_pid E_mother px_mother py_mother pz_mother" << std::endl;

  // Define counter for tau particles
  int tauCount = 0;
  int eventNumber = 0;
  double weight = 1.0; // Default weight for events without specific tau production
  double total_weight = 0.0;


  // Define the weights map once (outside the event loop, ideally)
  const std::unordered_map<int, double> motherWeights = {
      {411,     1.20e-3},
      {431,     5.36e-2},
      {100443,  3.1e-3}
  };

  // Main event generation loop
  while (tauCount < nTauEvents) {
    if (!pythia.next()) continue;
    
    // Soft event above pTcut
    if (soft_or_hard == "soft" && pythia.info.pTHat() > pTcut_d) continue; 
    
    // Hard event below pTcut
    if (soft_or_hard == "hard" && pythia.info.pTHat() < pTcut_d) continue; 

    // Check if the event contains any tau particles
    for (int i = 0; i < pythia.event.size(); ++i) {
      int id = pythia.event[i].id();
    
      // Tau found -- save to file
      if (abs(id) == 15) {

          int motherID = std::abs(pythia.event[pythia.event[i].mother1()].id());
          auto it = motherWeights.find(motherID);
          weight = (it != motherWeights.end()) ? it->second : 1.0;
          
          ++tauCount;
          total_weight += weight;

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
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].e()  : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].px() : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].py() : 0) << " " 
            << (pythia.event[i].mother1() > 0 ? pythia.event[pythia.event[i].mother1()].pz() : 0) << std::endl;
          
      }
    }

    // Increase counter of valid events (tau or no tau, but ok kinematics)
    ++eventNumber;

  }
  // Close the output file
  outFile.close();

  // Save the total cross section to a separate file
  double totalCrossSection = pythia.info.sigmaGen();
  std::string outputXsecFileName = "pythia8_events/xsec_" + soft_or_hard + "_" + baseFilename + ".txt";
  std::ofstream crossSectionFile(outputXsecFileName);

  if (crossSectionFile.is_open()) {
      crossSectionFile << "Total cross section of proton-proton collisions: " << totalCrossSection << " mb" << std::endl;
      crossSectionFile << "Total tau production cross section: " << total_weight/eventNumber * totalCrossSection << " mb" << std::endl;
      crossSectionFile.close();
  } else {
      std::cerr << "Error: Could not open file to save cross section information." << std::endl;
  }

  std::cout << "Total cross section of proton-proton collisions: " << totalCrossSection << " mb" << std::endl;


  return 0;
}
